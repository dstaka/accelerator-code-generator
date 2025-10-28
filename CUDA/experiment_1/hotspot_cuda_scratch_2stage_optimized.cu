#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Scratch, from-scratch CUDA GEMM implementations (no CUTLASS/cuBLAS)
// - WMMA Tensor Core GEMM: FP16 inputs, FP32 accumulate
// - Naive GEMM (for validation / fallback)

// Layout assumptions:
// - A: row-major [M, K]
// - B: column-major [K, N] (better for WMMA col-major load of B)
// - C: row-major [M, N]

// WMMA kernel: 16x16x16 tiles
__global__ void wmma_gemm_fp16_row_col(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
#if (__CUDA_ARCH__ >= 800)
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    int tiles_m = M / WMMA_M;
    int tiles_n = N / WMMA_N;
    int total_tiles = tiles_m * tiles_n;

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += (gridDim.x * blockDim.x / 32)) {
        int tm = tile_idx / tiles_n;
        int tn = tile_idx % tiles_n;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < K; k0 += WMMA_K) {
            const half* a_ptr = A + (tm*WMMA_M)*K + k0;
            const half* b_ptr = B + k0*N + (tn*WMMA_N); // B is column-major: leading dim = N
            wmma::load_matrix_sync(a_frag, a_ptr, K);
            wmma::load_matrix_sync(b_frag, b_ptr, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* c_ptr = C + (tm*WMMA_M)*N + (tn*WMMA_N);
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    }
#else
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#endif
}

// Naive GEMM in FP32 accumulation
__global__ void naive_gemm_fp16_row_row(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        float a = __half2float(A[row * K + k]);
        float b = __half2float(B[k * N + col]); // row-major B
        acc += a * b;
    }
    C[row * N + col] = acc;
}

extern "C" void launch_hotspot_cuda_scratch_wmma_fp16(
    const void* A, const void* B_colmajor, void* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    // Round to multiples of 16
    auto round16 = [](int x) { return (x + 15) & ~15; };
    M = round16(M); N = round16(N); K = round16(K);

    dim3 block(128);
    dim3 grid((M/16)*(N/16));
    if (grid.x == 0) grid.x = 1;
    wmma_gemm_fp16_row_col<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(A),
        reinterpret_cast<const half*>(B_colmajor),
        reinterpret_cast<float*>(C),
        M, N, K
    );
}

extern "C" void launch_hotspot_cuda_scratch_naive_fp16(
    const void* A, const void* B_rowmajor, void* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
    naive_gemm_fp16_row_row<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(A),
        reinterpret_cast<const half*>(B_rowmajor),
        reinterpret_cast<float*>(C),
        M, N, K
    );
}

// ===================== SMEM tiled + 2-stage pipeline (WMMA) =====================

// Block-tiled GEMM with shared-memory double buffering. A: row-major, B: col-major, C: row-major
// Tile sizes chosen for A100: BM=128, BN=128, BK=128 (as 8 WMMA steps of 16)
template<int BM, int BN, int BK>
__global__ void wmma_gemm_fp16_row_col_smem2stage(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
#if (__CUDA_ARCH__ >= 800)
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    static_assert(BK % WMMA_K == 0, "BK must be multiple of 16");

    // Block tile coordinates
    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;

    // Shared memory for double buffer: As[2][BM*BK], Bs[2][BK*BN]
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = As + 2 * (BM * BK);

    // Accumulator fragments for each warp tile
    // One warp computes a 16x16 tile; number of warp tiles per block: (BM/16) x (BN/16)
    int warp_id = (threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_row = BN / WMMA_N; // along N
    int tile_m_warp = warp_id / warps_per_row; // [0, BM/16)
    int tile_n_warp = warp_id % warps_per_row; // [0, BN/16)

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Global K traversal in steps of BK
    int stages = 2;
    int num_k_tiles = (K + BK - 1) / BK;

    // Helper lambdas to load A, B tiles for a given k-tile into shared buffers
    auto load_AB_tile = [&](int stage, int kt) {
        int k_begin = kt * BK;
        // Cooperative load A tile [BM x BK] row-major
        int t = threadIdx.x;
        int elemsA = BM * BK;
        for (int i = t; i < elemsA; i += blockDim.x) {
            int r = i / BK;
            int c = i % BK;
            int gm = block_m + r;
            int gk = k_begin + c;
            half v = __float2half(0.f);
            if (gm < M && gk < K) v = A[gm * K + gk];
            As[stage * elemsA + i] = v;
        }
        // Cooperative load B tile [BK x BN] column-major: element(k, n) at B[k + n*K]
        int elemsB = BK * BN;
        for (int i = t; i < elemsB; i += blockDim.x) {
            int r = i / BN; // k within tile
            int c = i % BN; // n within tile
            int gk = k_begin + r;
            int gn = block_n + c;
            half v = __float2half(0.f);
            if (gk < K && gn < N) v = B[gk + gn * K];
            Bs[stage * elemsB + i] = v;
        }
    };

    // Preload stage 0
    if (num_k_tiles > 0) {
        load_AB_tile(0, 0);
    }
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int cur = kt % stages;
        int nxt = (kt + 1) % stages;

        // Launch preload for next tile while computing current tile
        if (kt + 1 < num_k_tiles) {
            load_AB_tile(nxt, kt + 1);
        }

        // Compute on current stage from shared memory in WMMA_K steps
        half* As_cur = As + cur * (BM * BK);
        half* Bs_cur = Bs + cur * (BK * BN);

        for (int kk = 0; kk < BK; kk += WMMA_K) {
            // Each warp loads its A/B fragments from shared memory
            const half* a_tile = As_cur + (tile_m_warp * WMMA_M) * BK + kk;
            const half* b_tile = Bs_cur + kk * BN + (tile_n_warp * WMMA_N);

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, a_tile, BK);
            wmma::load_matrix_sync(b_frag, b_tile, BN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads(); // Ensure next stage loads are visible before switching
    }

    // Store accumulator to C
    int gm0 = block_m + tile_m_warp * WMMA_M;
    int gn0 = block_n + tile_n_warp * WMMA_N;
    if (gm0 < M && gn0 < N) {
        float* c_ptr = C + gm0 * N + gn0;
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    }
#else
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#endif
}

extern "C" void launch_hotspot_cuda_scratch_wmma_fp16_smem2stage(
    const void* A, const void* B_colmajor, void* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 block(256); // 8 warps per block => supports BM=128, BN=128 (64 warp tiles)
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    size_t smem_bytes = (2 * 128 * 128 + 2 * 128 * 128) * sizeof(half); // 2*(BM*BK) + 2*(BK*BN)
    wmma_gemm_fp16_row_col_smem2stage<128,128,128><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half*>(A),
        reinterpret_cast<const half*>(B_colmajor),
        reinterpret_cast<float*>(C),
        M, N, K
    );
}



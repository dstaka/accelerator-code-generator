#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

struct Args {
    int M{4096};          // q_len per head
    int L{8192};          // gathered kv length
    int D{128};           // head_dim_k (K of GEMM)
    int Dv{128};          // head_dim_v
    int total_k{65536};   // kv cache length to sample from
    int iters{50};
    int warmup{10};
    float softmax_scale{1.0f / std::sqrt(128.0f)}; // default for D=128
    std::string dtype{"fp16"};
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc >= 8) {
        a.M = std::atoi(argv[1]);
        a.L = std::atoi(argv[2]);
        a.D = std::atoi(argv[3]);
        a.Dv = std::atoi(argv[4]);
        a.iters = std::atoi(argv[5]);
        a.warmup = std::atoi(argv[6]);
        a.total_k = std::atoi(argv[7]);
    }
    if (argc >= 9) a.dtype = argv[8];
    if (argc >= 10) a.softmax_scale = std::atof(argv[9]);
    return a;
}

template <typename T>
__global__ void init_random_kernel(T* ptr, size_t n, unsigned int seed) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int x = seed ^ (unsigned int)i;
        x = 1664525u * x + 1013904223u;
        float r = (x & 0x00FFFFFF) / float(0x01000000);
        ptr[i] = (T)(r - 0.5f);
    }
}

// Gather rows of src [total_k x dim] into dst [L x dim] using indices[L]. If idx==-1, zero-fill
template <typename T>
__global__ void gather_rows(const T* __restrict__ src, int total_k, int ld_src,
                            const int* __restrict__ indices, int L,
                            T* __restrict__ dst, int ld_dst, int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < L) {
        int idx = indices[row];
        const T* src_row = (idx >= 0 && idx < total_k) ? (src + (size_t)idx * ld_src) : nullptr;
        T* dst_row = dst + (size_t)row * ld_dst;
        for (int j = 0; j < dim; ++j) {
            dst_row[j] = src_row ? src_row[j] : (T)0;
        }
    }
}

// Build validity mask from indices (1 if valid else 0)
__global__ void indices_to_mask(const int* __restrict__ indices, int L, unsigned char* __restrict__ mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) mask[i] = (indices[i] >= 0) ? 1 : 0;
}

// Row-wise scaled softmax with mask on invalid columns. P is [M x L] float32 in row-major
__global__ void scaled_masked_softmax(float* __restrict__ P, int M, int L, float scale, const unsigned char* __restrict__ valid_mask) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float* prow = P + (size_t)row * L;
    float mval = -INFINITY;
    for (int j = 0; j < L; ++j) {
        float v = valid_mask[j] ? (prow[j] * scale) : -INFINITY;
        mval = fmaxf(mval, v);
    }
    float s = 0.f;
    for (int j = 0; j < L; ++j) {
        float v = valid_mask[j] ? expf(prow[j] * scale - mval) : 0.f;
        prow[j] = v;
        s += v;
    }
    float invs = s > 0.f ? (1.f / s) : 0.f;
    for (int j = 0; j < L; ++j) prow[j] *= invs;
}

// FP32 GEMM for PV: C[M,N] = A[M,K] @ B[K,N]
static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 64;
static constexpr int BLOCK_K = 32;

__global__ void gemm_rr_fp32_fp32_fp32(const float* __restrict__ A, int M, int K,
                                       const float* __restrict__ B, int N, int Kb,
                                       float* __restrict__ C, int Nout) {
    extern __shared__ unsigned char smem[];
    float* As = reinterpret_cast<float*>(smem);
    float* Bs = reinterpret_cast<float*>(smem + BLOCK_M * BLOCK_K * sizeof(float));

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int rowBase = blockRow * BLOCK_M;
    int colBase = blockCol * BLOCK_N;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int cRow0 = rowBase + ty * 4;
    int cCol0 = colBase + tx * 4;

    float acc[4][4] = {};

    for (int kb = 0; kb < K; kb += BLOCK_K) {
        for (int i = ty; i < BLOCK_M; i += blockDim.y) {
            for (int j = tx; j < BLOCK_K; j += blockDim.x) {
                int ai = rowBase + i;
                int ak = kb + j;
                As[i * BLOCK_K + j] = (ai < M && ak < K) ? A[(size_t)ai * K + ak] : 0.0f;
            }
        }
        for (int i = ty; i < BLOCK_K; i += blockDim.y) {
            for (int j = tx; j < BLOCK_N; j += blockDim.x) {
                int bk = kb + i;
                int bj = colBase + j;
                Bs[i * BLOCK_N + j] = (bk < Kb && bj < N) ? B[(size_t)bk * N + bj] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            float af[4] = {
                As[(ty*4 + 0) * BLOCK_K + kk],
                As[(ty*4 + 1) * BLOCK_K + kk],
                As[(ty*4 + 2) * BLOCK_K + kk],
                As[(ty*4 + 3) * BLOCK_K + kk]
            };
            float bf[4] = {
                Bs[kk * BLOCK_N + (tx*4 + 0)],
                Bs[kk * BLOCK_N + (tx*4 + 1)],
                Bs[kk * BLOCK_N + (tx*4 + 2)],
                Bs[kk * BLOCK_N + (tx*4 + 3)]
            };
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += af[i] * bf[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r = cRow0 + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c = cCol0 + j;
                if (c < Nout) C[(size_t)r * Nout + c] = acc[i][j];
            }
        }
    }
}

// QK^T using WMMA Tensor Cores
// Compute P[M,L] = Q[M,D] @ (K_sel[L,D])^T
// A: row-major (M x D), B: treat as col-major (D x L) aliasing K_sel row-major (L x D) with ldb=D
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WARPS_PER_BLOCK = 8;       // 256 threads
static constexpr int WARP_TILES_M = 2;          // 2 x 16 = 32 rows per block
static constexpr int WARP_TILES_N = 4;          // 4 x 16 = 64 cols per block

__global__ void qkt_wmma_kernel(const half* __restrict__ Q, const half* __restrict__ K_sel,
                                float* __restrict__ P,
                                int M, int L, int D) {
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpRow = warpId / WARP_TILES_N;
    int warpCol = warpId % WARP_TILES_N;

    int blockRowBase = blockIdx.y * (WMMA_M * WARP_TILES_M);
    int blockColBase = blockIdx.x * (WMMA_N * WARP_TILES_N);
    int cRow = blockRowBase + warpRow * WMMA_M;
    int cCol = blockColBase + warpCol * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag; // treat K^T
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < D; k0 += WMMA_K) {
        const half* a_ptr = Q + (size_t)cRow * D + k0;           // row-major, lda=D
        const half* b_ptr = K_sel + (size_t)0;                   // we will index via load stride
        // For B col-major (K=D, N=L) with ldb=D, base element at (k0, cCol)
        b_ptr = K_sel + (size_t)cCol * D + k0;                   // K_sel is (L x D) row-major -> (D x L) col-major alias

        wmma::load_matrix_sync(a_frag, a_ptr, D);
        wmma::load_matrix_sync(b_frag, b_ptr, D);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_ptr = P + (size_t)cRow * L + cCol;
    wmma::store_matrix_sync(c_ptr, c_frag, L, wmma::mem_row_major);
}

// Cast FP16 -> FP32
__global__ void cast_half_to_float(const __half* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::string d = args.dtype; for (auto &c : d) c = (char)std::tolower(c);
    std::printf("A100 KV-cache attention benchmark (scratch WMMA)\n");
    std::printf("M=%d L=%d D=%d Dv=%d total_k=%d iters=%d warmup=%d dtype=%s scale=%.6f\n",
                args.M, args.L, args.D, args.Dv, args.total_k, args.iters, args.warmup, d.c_str(), args.softmax_scale);

    if ((args.M % 16) || (args.L % 16) || (args.D % 16)) {
        std::fprintf(stderr, "[WARN] WMMA path requires M,L,D to be multiples of 16 for correctness.\n");
    }

    checkCuda(cudaSetDevice(0), "set device");

    const int M = args.M, L = args.L, D = args.D, Dv = args.Dv, TK = args.total_k;
    cudaEvent_t start, stop; checkCuda(cudaEventCreate(&start), "ev start"); checkCuda(cudaEventCreate(&stop), "ev stop");

    // Host-side indices
    std::vector<int> h_idx(L);
    for (int i = 0; i < L; ++i) {
        int r = rand();
        if ((r & 31) == 0) h_idx[i] = -1; else h_idx[i] = r % TK;
    }
    int* d_idx = nullptr; checkCuda(cudaMalloc(&d_idx, L * sizeof(int)), "malloc idx");
    checkCuda(cudaMemcpy(d_idx, h_idx.data(), L * sizeof(int), cudaMemcpyHostToDevice), "copy idx");
    unsigned char* d_mask = nullptr; checkCuda(cudaMalloc(&d_mask, L * sizeof(unsigned char)), "malloc mask");
    indices_to_mask<<<(L+255)/256, 256>>>(d_idx, L, d_mask);
    checkCuda(cudaDeviceSynchronize(), "mask sync");

    if (d == "fp16" || d == "half") {
        using T = __half;
        // Q[M x D], K_total[TK x D], V_total[TK x Dv]
        T *Q=nullptr, *K_total=nullptr, *V_total=nullptr;
        T *K_sel=nullptr, *V_sel=nullptr; // [L x D], [L x Dv]
        float *P=nullptr; // [M x L]
        float *O=nullptr; // [M x Dv]
        float *V_sel_f32=nullptr; // [L x Dv]
        checkCuda(cudaMalloc(&Q, (size_t)M*D*sizeof(T)), "malloc Q");
        checkCuda(cudaMalloc(&K_total, (size_t)TK*D*sizeof(T)), "malloc K");
        checkCuda(cudaMalloc(&V_total, (size_t)TK*Dv*sizeof(T)), "malloc V");
        checkCuda(cudaMalloc(&K_sel, (size_t)L*D*sizeof(T)), "malloc K_sel");
        checkCuda(cudaMalloc(&V_sel, (size_t)L*Dv*sizeof(T)), "malloc V_sel");
        checkCuda(cudaMalloc(&P, (size_t)M*L*sizeof(float)), "malloc P");
        checkCuda(cudaMalloc(&O, (size_t)M*Dv*sizeof(float)), "malloc O");
        checkCuda(cudaMalloc(&V_sel_f32, (size_t)L*Dv*sizeof(float)), "malloc V_sel_f32");

        int threads=256; 
        int bQ=(int)(((size_t)M*D + threads-1)/threads);
        int bK=(int)(((size_t)TK*D + threads-1)/threads);
        int bV=(int)(((size_t)TK*Dv + threads-1)/threads);
        init_random_kernel<<<bQ,threads>>>(Q, (size_t)M*D, 111u);
        init_random_kernel<<<bK,threads>>>(K_total, (size_t)TK*D, 222u);
        init_random_kernel<<<bV,threads>>>(V_total, (size_t)TK*Dv, 333u);
        checkCuda(cudaDeviceSynchronize(), "init sync");

        // Launch configs
        dim3 grid_qk((L + (WMMA_N*WARP_TILES_N) - 1) / (WMMA_N*WARP_TILES_N),
                     (M + (WMMA_M*WARP_TILES_M) - 1) / (WMMA_M*WARP_TILES_M));
        dim3 block_qk(32 * WARPS_PER_BLOCK, 1, 1);
        dim3 block_pv(16, 16);
        dim3 grid_pv((Dv + BLOCK_N - 1)/BLOCK_N, (M + BLOCK_M - 1)/BLOCK_M);
        size_t smem_pv = (size_t)BLOCK_M*BLOCK_K*sizeof(float) + (size_t)BLOCK_K*BLOCK_N*sizeof(float);

        // Create K_sel, V_sel
        gather_rows<<<(L+255)/256, 256>>>(K_total, TK, D, d_idx, L, K_sel, D, D);
        gather_rows<<<(L+255)/256, 256>>>(V_total, TK, Dv, d_idx, L, V_sel, Dv, Dv);
        checkCuda(cudaDeviceSynchronize(), "gather sync");

        // Warmup
        for (int i = 0; i < args.warmup; ++i) {
            // QK^T on WMMA
            qkt_wmma_kernel<<<grid_qk, block_qk>>>(Q, K_sel, P, M, L, D);
            checkCuda(cudaGetLastError(), "qk wmma");
            // softmax
            scaled_masked_softmax<<<(M+255)/256, 256>>>(P, M, L, args.softmax_scale, d_mask);
            // PV: cast V to f32, then P[M,L] x V[L,Dv] in FP32
            cast_half_to_float<<<(size_t(L)*Dv + 255)/256, 256>>>(V_sel, V_sel_f32, (size_t)L*Dv);
            gemm_rr_fp32_fp32_fp32<<<grid_pv, block_pv, smem_pv>>>(P, M, L, V_sel_f32, Dv, L, O, Dv);
            checkCuda(cudaGetLastError(), "pv gemm");
        }

        checkCuda(cudaEventRecord(start), "record start");
        for (int i = 0; i < args.iters; ++i) {
            // Regather per-iter to mirror cuBLAS bench behavior
            gather_rows<<<(L+255)/256, 256>>>(K_total, TK, D, d_idx, L, K_sel, D, D);
            gather_rows<<<(L+255)/256, 256>>>(V_total, TK, Dv, d_idx, L, V_sel, Dv, Dv);

            qkt_wmma_kernel<<<grid_qk, block_qk>>>(Q, K_sel, P, M, L, D);
            scaled_masked_softmax<<<(M+255)/256, 256>>>(P, M, L, args.softmax_scale, d_mask);
            cast_half_to_float<<<(size_t(L)*Dv + 255)/256, 256>>>(V_sel, V_sel_f32, (size_t)L*Dv);
            gemm_rr_fp32_fp32_fp32<<<grid_pv, block_pv, smem_pv>>>(P, M, L, V_sel_f32, Dv, L, O, Dv);
        }
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");
        float ms=0.f; checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed"); ms/=args.iters;
        // FLOPs: QK^T (2*M*L*D) + PV (2*M*Dv*L)
        double flops = 2.0*(double)M*L*(D + Dv);
        double tflops = flops/(ms*1.0e9);
        std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

        cudaFree(Q); cudaFree(K_total); cudaFree(V_total);
        cudaFree(K_sel); cudaFree(V_sel); cudaFree(P); cudaFree(O); cudaFree(V_sel_f32);
    } else {
        std::fprintf(stderr, "Only fp16 is implemented in this sample.\n");
        return 1;
    }

    cudaFree(d_idx); cudaFree(d_mask);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

// nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -o a100_kvcache_bench_scratch_wmma a100_kvcache_bench_scratch_wmma.cu
// ./a100_kvcache_bench_scratch_wmma 4096 8192 128 128 50 10 65536 fp16

// Log
// $ ./a100_kvcache_bench_scratch_wmma 4096 8192 128 128 50 10 65536 fp16
// A100 KV-cache attention benchmark (scratch WMMA)
// M=4096 L=8192 D=128 Dv=128 total_k=65536 iters=50 warmup=10 dtype=fp16 scale=0.088388
// Average time: 12.661 ms, Throughput: 1.36 TFLOPs
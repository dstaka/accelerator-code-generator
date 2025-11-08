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
    int M{4096};
    int L{8192};
    int D{128};
    int Dv{128};
    int total_k{65536};
    int iters{50};
    int warmup{10};
    float softmax_scale{1.0f / std::sqrt(128.0f)};
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

__global__ void indices_to_mask(const int* __restrict__ indices, int L, unsigned char* __restrict__ mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) mask[i] = (indices[i] >= 0) ? 1 : 0;
}

// Warp-level parallel softmax reduction
__global__ void warp_scaled_masked_softmax(float* __restrict__ P, int M, int L, float scale, const unsigned char* __restrict__ valid_mask) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int lane = threadIdx.x % 32;
    float* prow = P + (size_t)row * L;

    // Find max with warp reduction
    float mval = -INFINITY;
    for (int j = lane; j < L; j += 32) {
        float v = valid_mask[j] ? (prow[j] * scale) : -INFINITY;
        mval = fmaxf(mval, v);
    }
    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2) {
        mval = fmaxf(mval, __shfl_down_sync(0xffffffff, mval, offset));
    }
    mval = __shfl_sync(0xffffffff, mval, 0);

    // Compute exp and sum with warp reduction
    float s = 0.f;
    for (int j = lane; j < L; j += 32) {
        float v = valid_mask[j] ? expf(prow[j] * scale - mval) : 0.f;
        prow[j] = v;
        s += v;
    }
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        s += __shfl_down_sync(0xffffffff, s, offset);
    }
    s = __shfl_sync(0xffffffff, s, 0);

    // Normalize
    float invs = s > 0.f ? (1.f / s) : 0.f;
    for (int j = lane; j < L; j += 32) {
        prow[j] *= invs;
    }
}

// QK^T using WMMA
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int WARP_TILES_M = 2;
static constexpr int WARP_TILES_N = 4;

__global__ void qkt_wmma_kernel(const half* __restrict__ Q, const half* __restrict__ K_sel,
                                float* __restrict__ P, int M, int L, int D) {
    int warpId = threadIdx.x / 32;
    int warpRow = warpId / WARP_TILES_N;
    int warpCol = warpId % WARP_TILES_N;

    int blockRowBase = blockIdx.y * (WMMA_M * WARP_TILES_M);
    int blockColBase = blockIdx.x * (WMMA_N * WARP_TILES_N);
    int cRow = blockRowBase + warpRow * WMMA_M;
    int cCol = blockColBase + warpCol * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < D; k0 += WMMA_K) {
        const half* a_ptr = Q + (size_t)cRow * D + k0;
        const half* b_ptr = K_sel + (size_t)cCol * D + k0;
        wmma::load_matrix_sync(a_frag, a_ptr, D);
        wmma::load_matrix_sync(b_frag, b_ptr, D);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_ptr = P + (size_t)cRow * L + cCol;
    wmma::store_matrix_sync(c_ptr, c_frag, L, wmma::mem_row_major);
}

// Optimized FP32 GEMM for PV with larger tiles
static constexpr int BLOCK_M = 128;
static constexpr int BLOCK_N = 128;
static constexpr int BLOCK_K = 16;
static constexpr int THREAD_M = 8;
static constexpr int THREAD_N = 8;

__global__ void gemm_pv_optimized(const float* __restrict__ P, const float* __restrict__ V_sel,
                                  float* __restrict__ O, int M, int L, int Dv) {
    __shared__ float Ps[BLOCK_M][BLOCK_K];
    __shared__ float Vs[BLOCK_K][BLOCK_N];

    int tx = threadIdx.x % 16;
    int ty = threadIdx.y % 8;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int rowBase = by * BLOCK_M;
    int colBase = bx * BLOCK_N;

    float acc[THREAD_M][THREAD_N] = {};

    for (int k = 0; k < L; k += BLOCK_K) {
        // Load P tile
        for (int i = ty; i < BLOCK_M; i += 8) {
            for (int j = tx; j < BLOCK_K; j += 16) {
                int row = rowBase + i;
                int col = k + j;
                Ps[i][j] = (row < M && col < L) ? P[(size_t)row * L + col] : 0.0f;
            }
        }

        // Load V tile (already FP32, no cast needed)
        for (int i = ty; i < BLOCK_K; i += 8) {
            for (int j = tx; j < BLOCK_N; j += 16) {
                int row = k + i;
                int col = colBase + j;
                Vs[i][j] = (row < L && col < Dv) ? V_sel[(size_t)row * Dv + col] : 0.0f;
            }
        }
        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] += Ps[ty * THREAD_M + i][kk] * Vs[kk][tx * THREAD_N + j];
                }
            }
        }
        __syncthreads();
    }

    // Store results
    for (int i = 0; i < THREAD_M; ++i) {
        for (int j = 0; j < THREAD_N; ++j) {
            int row = rowBase + ty * THREAD_M + i;
            int col = colBase + tx * THREAD_N + j;
            if (row < M && col < Dv) {
                O[(size_t)row * Dv + col] = acc[i][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::string d = args.dtype; for (auto &c : d) c = (char)std::tolower(c);
    std::printf("A100 KV-cache attention benchmark (v1.2.0 - Warp Softmax + WMMA, V in FP32)\n");
    std::printf("M=%d L=%d D=%d Dv=%d total_k=%d iters=%d warmup=%d dtype=%s scale=%.6f\n",
                args.M, args.L, args.D, args.Dv, args.total_k, args.iters, args.warmup, d.c_str(), args.softmax_scale);

    if ((args.M % 16) || (args.L % 16) || (args.D % 16)) {
        std::fprintf(stderr, "[WARN] WMMA requires M,L,D to be multiples of 16.\n");
    }

    checkCuda(cudaSetDevice(0), "set device");

    const int M = args.M, L = args.L, D = args.D, Dv = args.Dv, TK = args.total_k;
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "ev start");
    checkCuda(cudaEventCreate(&stop), "ev stop");

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
        T *Q=nullptr, *K_total=nullptr;
        T *K_sel=nullptr;
        float *V_total=nullptr, *V_sel=nullptr; // V is FP32 for fair comparison
        float *P=nullptr;
        float *O=nullptr;

        checkCuda(cudaMalloc(&Q, (size_t)M*D*sizeof(T)), "malloc Q");
        checkCuda(cudaMalloc(&K_total, (size_t)TK*D*sizeof(T)), "malloc K");
        checkCuda(cudaMalloc(&V_total, (size_t)TK*Dv*sizeof(float)), "malloc V");
        checkCuda(cudaMalloc(&K_sel, (size_t)L*D*sizeof(T)), "malloc K_sel");
        checkCuda(cudaMalloc(&V_sel, (size_t)L*Dv*sizeof(float)), "malloc V_sel");
        checkCuda(cudaMalloc(&P, (size_t)M*L*sizeof(float)), "malloc P");
        checkCuda(cudaMalloc(&O, (size_t)M*Dv*sizeof(float)), "malloc O");

        int threads=256;
        int bQ=(int)(((size_t)M*D + threads-1)/threads);
        int bK=(int)(((size_t)TK*D + threads-1)/threads);
        int bV=(int)(((size_t)TK*Dv + threads-1)/threads);
        init_random_kernel<<<bQ,threads>>>(Q, (size_t)M*D, 111u);
        init_random_kernel<<<bK,threads>>>(K_total, (size_t)TK*D, 222u);
        init_random_kernel<<<bV,threads>>>(V_total, (size_t)TK*Dv, 333u);  // V_total is now float
        checkCuda(cudaDeviceSynchronize(), "init sync");

        dim3 grid_qk((L + (WMMA_N*WARP_TILES_N) - 1) / (WMMA_N*WARP_TILES_N),
                     (M + (WMMA_M*WARP_TILES_M) - 1) / (WMMA_M*WARP_TILES_M));
        dim3 block_qk(32 * WARPS_PER_BLOCK, 1, 1);

        dim3 block_pv(16, 8);
        dim3 grid_pv((Dv + BLOCK_N - 1)/BLOCK_N, (M + BLOCK_M - 1)/BLOCK_M);

        gather_rows<<<(L+255)/256, 256>>>(K_total, TK, D, d_idx, L, K_sel, D, D);
        gather_rows<<<(L+255)/256, 256>>>(V_total, TK, Dv, d_idx, L, V_sel, Dv, Dv);
        checkCuda(cudaDeviceSynchronize(), "gather sync");

        // Warmup
        for (int i = 0; i < args.warmup; ++i) {
            qkt_wmma_kernel<<<grid_qk, block_qk>>>(Q, K_sel, P, M, L, D);
            warp_scaled_masked_softmax<<<(M+31)/32, 32>>>(P, M, L, args.softmax_scale, d_mask);
            gemm_pv_optimized<<<grid_pv, block_pv>>>(P, V_sel, O, M, L, Dv);
        }
        checkCuda(cudaDeviceSynchronize(), "warmup sync");

        checkCuda(cudaEventRecord(start), "record start");
        for (int i = 0; i < args.iters; ++i) {
            gather_rows<<<(L+255)/256, 256>>>(K_total, TK, D, d_idx, L, K_sel, D, D);
            gather_rows<<<(L+255)/256, 256>>>(V_total, TK, Dv, d_idx, L, V_sel, Dv, Dv);
            qkt_wmma_kernel<<<grid_qk, block_qk>>>(Q, K_sel, P, M, L, D);
            warp_scaled_masked_softmax<<<(M+31)/32, 32>>>(P, M, L, args.softmax_scale, d_mask);
            gemm_pv_optimized<<<grid_pv, block_pv>>>(P, V_sel, O, M, L, Dv);
        }
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");

        float ms=0.f; checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed"); ms/=args.iters;
        double flops = 2.0*(double)M*L*(D + Dv);
        double tflops = flops/(ms*1.0e9);
        std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

        cudaFree(Q); cudaFree(K_total); cudaFree(V_total);
        cudaFree(K_sel); cudaFree(V_sel); cudaFree(P); cudaFree(O);
    } else {
        std::fprintf(stderr, "Only fp16 is implemented.\n");
        return 1;
    }

    cudaFree(d_idx); cudaFree(d_mask);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}



// nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -lcublas -o a100_kvcache_bench_scratch_wmma a100_kvcache_bench_scratch_wmma.cu
// ./a100_kvcache_bench_scratch_wmma 4096 8192 128 128 50 10 65536 fp16

// Log (v1.2.0 - Fair comparison with V in FP32)
// ./a100_kvcache_bench_scratch_wmma 4096 8192 128 128 50 10 65536 fp16
// A100 KV-cache attention benchmark (v1.2.0 - Warp Softmax + WMMA, V in FP32)
// M=4096 L=8192 D=128 Dv=128 total_k=65536 iters=50 warmup=10 dtype=fp16 scale=0.088388
// [Performance will be measured after recompilation]

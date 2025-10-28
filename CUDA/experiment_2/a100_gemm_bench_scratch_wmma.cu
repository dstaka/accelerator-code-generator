#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
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
    int M{16384};
    int N{16384};
    int K{16384};
    int iters{50};
    int warmup{10};
    std::string dtype{"fp16"}; // fp16 only here
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc >= 6) { a.M = std::atoi(argv[1]); a.N = std::atoi(argv[2]); a.K = std::atoi(argv[3]); a.iters = std::atoi(argv[4]); a.warmup = std::atoi(argv[5]); }
    if (argc >= 7) { a.dtype = argv[6]; }
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

// WMMA kernel: FP16 inputs, FP32 accumulate
// Requires M,N,K to be multiples of 16 for correctness
static constexpr int TILE_M = 16;
static constexpr int TILE_N = 16;
static constexpr int TILE_K = 16;
static constexpr int WARPS_PER_BLOCK = 8;            // 8 warps = 256 threads
static constexpr int WARP_TILES_M = 2;               // 2 x 16 = 32 rows per block
static constexpr int WARP_TILES_N = 4;               // 4 x 16 = 64 cols per block
static_assert(WARP_TILES_M * WARP_TILES_N == WARPS_PER_BLOCK, "warp tiling mismatch");

__global__ void gemm_wmma_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K,
                                 float alpha, float beta) {
    int warpId = threadIdx.x / 32;                 // 0..7
    int laneId = threadIdx.x % 32;                 // not used
    int warpRow = warpId / WARP_TILES_N;           // 0..1
    int warpCol = warpId % WARP_TILES_N;           // 0..3

    int blockRowBase = blockIdx.y * (TILE_M * WARP_TILES_M);
    int blockColBase = blockIdx.x * (TILE_N * WARP_TILES_N);

    int cRow = blockRowBase + warpRow * TILE_M;
    int cCol = blockColBase + warpCol * TILE_N;

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K in 16
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const half* a_ptr = A + (size_t)cRow * K + k0;
        const half* b_ptr = B + (size_t)k0 * N + cCol;
        // Assumes bounds are valid (multiples of 16)
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store C tile
    float* c_ptr = C + (size_t)cRow * N + cCol;
    if (beta == 0.0f) {
        // Scale alpha during store
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; ++i) {
            c_frag.x[i] = alpha * c_frag.x[i];
        }
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    } else {
        // Read-modify-write
        float tmp[TILE_M * TILE_N];
        wmma::store_matrix_sync(tmp, c_frag, TILE_N, wmma::mem_row_major);
        #pragma unroll
        for (int mi = 0; mi < TILE_M; ++mi) {
            int r = cRow + mi;
            if (r < M) {
                #pragma unroll
                for (int nj = 0; nj < TILE_N; ++nj) {
                    int c = cCol + nj;
                    if (c < N) {
                        float prev = C[(size_t)r * N + c];
                        C[(size_t)r * N + c] = alpha * tmp[mi * TILE_N + nj] + beta * prev;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::printf("A100 GEMM benchmark (scratch WMMA)\n");
    std::printf("M=%d N=%d K=%d iters=%d warmup=%d dtype=%s\n", args.M, args.N, args.K, args.iters, args.warmup, args.dtype.c_str());

    if ((args.M % 16) || (args.N % 16) || (args.K % 16)) {
        std::fprintf(stderr, "[WARN] WMMA path requires M,N,K to be multiples of 16 for correctness.\n");
    }

    int device = 0;
    checkCuda(cudaSetDevice(device), "set device");

    size_t numA = (size_t)args.M * args.K;
    size_t numB = (size_t)args.K * args.N;
    size_t numC = (size_t)args.M * args.N;

    half *A{nullptr}, *B{nullptr};
    float *C{nullptr};
    checkCuda(cudaMalloc(&A, numA * sizeof(half)), "malloc A");
    checkCuda(cudaMalloc(&B, numB * sizeof(half)), "malloc B");
    checkCuda(cudaMalloc(&C, numC * sizeof(float)), "malloc C");

    int threads = 256;
    int blocksA = (int)((numA + threads - 1) / threads);
    int blocksB = (int)((numB + threads - 1) / threads);
    init_random_kernel<<<blocksA, threads>>>(A, numA, 12345u);
    init_random_kernel<<<blocksB, threads>>>(B, numB, 67890u);
    checkCuda(cudaGetLastError(), "init kernels launch");
    checkCuda(cudaDeviceSynchronize(), "init sync");

    dim3 block(32 * WARPS_PER_BLOCK, 1, 1); // 256 threads
    dim3 grid((args.N + (TILE_N * WARP_TILES_N) - 1) / (TILE_N * WARP_TILES_N),
              (args.M + (TILE_M * WARP_TILES_M) - 1) / (TILE_M * WARP_TILES_M));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");

    const float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < args.warmup; ++i) {
        gemm_wmma_kernel<<<grid, block>>>(A, B, C, args.M, args.N, args.K, alpha, 0.0f);
    }

    checkCuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < args.iters; ++i) {
        gemm_wmma_kernel<<<grid, block>>>(A, B, C, args.M, args.N, args.K, alpha, 0.0f);
    }
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    ms /= (float)args.iters;
    double tflops = 2.0 * (double)args.M * args.N * args.K / (ms * 1.0e9);
    std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

    cudaFree(A); cudaFree(B); cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

// nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -o a100_gemm_bench_scratch_wmma a100_gemm_bench_scratch_wmma.cu
// ./a100_gemm_bench_scratch_wmma 16384 16384 16384 50 10 fp16

// Log
// $ ./a100_gemm_bench_scratch_wmma 16384 16384 16384 50 10 fp16
// A100 GEMM benchmark (scratch WMMA)
// M=16384 N=16384 K=16384 iters=50 warmup=10 dtype=fp16
// Average time: 531.169 ms, Throughput: 16.56 TFLOPs


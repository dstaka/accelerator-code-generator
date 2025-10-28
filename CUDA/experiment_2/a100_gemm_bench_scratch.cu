#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_fp16.h>

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// No cuBLAS used in scratch implementation

struct Args {
    int M{16384};
    int N{16384};
    int K{16384};
    int iters{50};
    int warmup{10};
    std::string dtype{"fp16"}; // fp16|bf16|tf32
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
        // simple LCG
        unsigned int x = seed ^ (unsigned int)i;
        x = 1664525u * x + 1013904223u;
        float r = (x & 0x00FFFFFF) / float(0x01000000);
        ptr[i] = (T)(r - 0.5f); // roughly [-0.5, 0.5)
    }
}

// Simple tiled GEMM kernel: FP16 inputs, FP32 accumulate/output
// C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C
static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 64;
static constexpr int BLOCK_K = 32;

__global__ void gemm_fp16_kernel(const __half* __restrict__ A,
                                 const __half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K,
                                 float alpha, float beta) {
    extern __shared__ unsigned char smem_raw[];
    __half* As = reinterpret_cast<__half*>(smem_raw);
    __half* Bs = reinterpret_cast<__half*>(smem_raw + BLOCK_M * BLOCK_K * sizeof(__half));

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int rowBase = blockRow * BLOCK_M;
    int colBase = blockCol * BLOCK_N;

    int ty = threadIdx.y; // 0..15
    int tx = threadIdx.x; // 0..15
    int cRow0 = rowBase + ty * 4;
    int cCol0 = colBase + tx * 4;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    }

    for (int kb = 0; kb < K; kb += BLOCK_K) {
        // Load A tile [BLOCK_M x BLOCK_K]
        for (int i = ty; i < BLOCK_M; i += blockDim.y) {
            for (int j = tx; j < BLOCK_K; j += blockDim.x) {
                int ai = rowBase + i;
                int ak = kb + j;
                As[i * BLOCK_K + j] = (ai < M && ak < K) ? A[(size_t)ai * K + ak] : __float2half(0.f);
            }
        }
        // Load B tile [BLOCK_K x BLOCK_N]
        for (int i = ty; i < BLOCK_K; i += blockDim.y) {
            for (int j = tx; j < BLOCK_N; j += blockDim.x) {
                int bk = kb + i;
                int bj = colBase + j;
                Bs[i * BLOCK_N + j] = (bk < K && bj < N) ? B[(size_t)bk * N + bj] : __float2half(0.f);
            }
        }
        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            __half a0 = As[(ty*4 + 0) * BLOCK_K + kk];
            __half a1 = As[(ty*4 + 1) * BLOCK_K + kk];
            __half a2 = As[(ty*4 + 2) * BLOCK_K + kk];
            __half a3 = As[(ty*4 + 3) * BLOCK_K + kk];
            float af[4] = {__half2float(a0), __half2float(a1), __half2float(a2), __half2float(a3)};

            __half b0 = Bs[kk * BLOCK_N + (tx*4 + 0)];
            __half b1 = Bs[kk * BLOCK_N + (tx*4 + 1)];
            __half b2 = Bs[kk * BLOCK_N + (tx*4 + 2)];
            __half b3 = Bs[kk * BLOCK_N + (tx*4 + 3)];
            float bf[4] = {__half2float(b0), __half2float(b1), __half2float(b2), __half2float(b3)};

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += af[i] * bf[j];
                }
            }
        }
        __syncthreads();
    }

    // Write back
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r = cRow0 + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c = cCol0 + j;
                if (c < N) {
                    if (beta == 0.0f) {
                        C[(size_t)r * N + c] = alpha * acc[i][j];
                    } else {
                        float prev = C[(size_t)r * N + c];
                        C[(size_t)r * N + c] = alpha * acc[i][j] + beta * prev;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::printf("A100 GEMM benchmark (scratch CUDA)\n");
    std::printf("M=%d N=%d K=%d iters=%d warmup=%d dtype=%s\n", args.M, args.N, args.K, args.iters, args.warmup, args.dtype.c_str());

    int device = 0;
    checkCuda(cudaSetDevice(device), "set device");

    // No cuBLAS handle

    // Column-major by default for cuBLAS: A (MxK), B (KxN), C (MxN)
    size_t numA = (size_t)args.M * args.K;
    size_t numB = (size_t)args.K * args.N;
    size_t numC = (size_t)args.M * args.N;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");

    const float alpha = 1.0f, beta = 0.0f;

    std::string d = args.dtype;
    for (auto &ch : d) ch = (char)std::tolower(ch);

    if (d == "fp16" || d == "half") {
        using T = __half;
        T *A{nullptr}, *B{nullptr};
        float *C{nullptr};
        checkCuda(cudaMalloc(&A, numA * sizeof(T)), "malloc A");
        checkCuda(cudaMalloc(&B, numB * sizeof(T)), "malloc B");
        checkCuda(cudaMalloc(&C, numC * sizeof(float)), "malloc C");
        int threads = 256;
        int blocksA = (int)((numA + threads - 1) / threads);
        int blocksB = (int)((numB + threads - 1) / threads);
        init_random_kernel<<<blocksA, threads>>>(A, numA, 12345u);
        init_random_kernel<<<blocksB, threads>>>(B, numB, 67890u);
        checkCuda(cudaGetLastError(), "init kernels launch");
        checkCuda(cudaDeviceSynchronize(), "init sync");
        // Launch configuration
        dim3 block(16, 16);
        dim3 grid((args.N + BLOCK_N - 1) / BLOCK_N, (args.M + BLOCK_M - 1) / BLOCK_M);
        size_t smem_sz = (size_t)BLOCK_M * BLOCK_K * sizeof(__half) + (size_t)BLOCK_K * BLOCK_N * sizeof(__half);

        // Warmup
        for (int i = 0; i < args.warmup; ++i) {
            gemm_fp16_kernel<<<grid, block, smem_sz>>>(A, B, C, args.M, args.N, args.K, alpha, 0.0f);
        }

        checkCuda(cudaEventRecord(start), "record start");
        for (int i = 0; i < args.iters; ++i) {
            gemm_fp16_kernel<<<grid, block, smem_sz>>>(A, B, C, args.M, args.N, args.K, alpha, 0.0f);
        }
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");
        float ms = 0.f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        ms /= (float)args.iters;
        double tflops = 2.0 * (double)args.M * args.N * args.K / (ms * 1.0e9);
        std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

        cudaFree(A); cudaFree(B); cudaFree(C);
    } else {
        std::fprintf(stderr, "Only fp16 is implemented in this scratch GEMM. Use dtype=fp16\n");
        return 1;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}


// nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -o a100_gemm_bench_scratch a100_gemm_bench_scratch.cu
// ./a100_gemm_bench_scratch 16384 16384 16384 50 10 fp16

// Log
// $ ./a100_gemm_bench_scratch 16384 16384 16384 50 10 fp16
// A100 GEMM benchmark (scratch CUDA)
// M=16384 N=16384 K=16384 iters=50 warmup=10 dtype=fp16
// Average time: 820.062 ms, Throughput: 10.73 TFLOPs
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

static void checkCublas(cublasStatus_t stat, const char* msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS Error %s: %d\n", msg, (int)stat);
        std::exit(EXIT_FAILURE);
    }
}

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
        unsigned int x = seed ^ (unsigned int)i;
        x = 1664525u * x + 1013904223u;
        float r = (x & 0x00FFFFFF) / float(0x01000000);
        ptr[i] = (T)(r - 0.5f);
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::printf("A100 GEMM benchmark (cuBLAS)\n");
    std::printf("M=%d N=%d K=%d iters=%d warmup=%d dtype=%s\n", args.M, args.N, args.K, args.iters, args.warmup, args.dtype.c_str());

    int device = 0;
    checkCuda(cudaSetDevice(device), "set device");

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "create handle");
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "set math mode");

    size_t numA = (size_t)args.M * args.K;
    size_t numB = (size_t)args.K * args.N;
    size_t numC = (size_t)args.M * args.N;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");

    const float alpha = 1.0f, beta = 0.0f;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;

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

        for (int i = 0; i < args.warmup; ++i) {
            checkCublas(cublasGemmEx(
                handle, transB, transA,
                args.N, args.M, args.K,
                &alpha,
                B, CUDA_R_16F, args.N,
                A, CUDA_R_16F, args.K,
                &beta,
                C, CUDA_R_32F, args.N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ), "gemm warmup");
        }

        checkCuda(cudaEventRecord(start), "record start");
        for (int i = 0; i < args.iters; ++i) {
            checkCublas(cublasGemmEx(
                handle, transB, transA,
                args.N, args.M, args.K,
                &alpha,
                B, CUDA_R_16F, args.N,
                A, CUDA_R_16F, args.K,
                &beta,
                C, CUDA_R_32F, args.N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ), "gemm");
        }
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");
        float ms = 0.f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        ms /= (float)args.iters;
        double tflops = 2.0 * (double)args.M * args.N * args.K / (ms * 1.0e9);
        std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

        cudaFree(A); cudaFree(B); cudaFree(C);
    } else if (d == "bf16" || d == "bfloat16") {
        using T = __nv_bfloat16;
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

        for (int i = 0; i < args.warmup; ++i) {
            checkCublas(cublasGemmEx(
                handle, transB, transA,
                args.N, args.M, args.K,
                &alpha,
                B, CUDA_R_16BF, args.N,
                A, CUDA_R_16BF, args.K,
                &beta,
                C, CUDA_R_32F, args.N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ), "gemm warmup");
        }

        checkCuda(cudaEventRecord(start), "record start");
        for (int i = 0; i < args.iters; ++i) {
            checkCublas(cublasGemmEx(
                handle, transB, transA,
                args.N, args.M, args.K,
                &alpha,
                B, CUDA_R_16BF, args.N,
                A, CUDA_R_16BF, args.K,
                &beta,
                C, CUDA_R_32F, args.N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ), "gemm");
        }
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");
        float ms = 0.f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        ms /= (float)args.iters;
        double tflops = 2.0 * (double)args.M * args.N * args.K / (ms * 1.0e9);
        std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

        cudaFree(A); cudaFree(B); cudaFree(C);
    } else { // tf32 (fp32 inputs, compute allow TF32)
        float *A{nullptr}, *B{nullptr}, *C{nullptr};
        checkCuda(cudaMalloc(&A, numA * sizeof(float)), "malloc A");
        checkCuda(cudaMalloc(&B, numB * sizeof(float)), "malloc B");
        checkCuda(cudaMalloc(&C, numC * sizeof(float)), "malloc C");
        int threads = 256;
        int blocksA = (int)((numA + threads - 1) / threads);
        int blocksB = (int)((numB + threads - 1) / threads);
        init_random_kernel<<<blocksA, threads>>>(A, numA, 12345u);
        init_random_kernel<<<blocksB, threads>>>(B, numB, 67890u);
        checkCuda(cudaGetLastError(), "init kernels launch");
        checkCuda(cudaDeviceSynchronize(), "init sync");

        checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "set TF32 math");

        for (int i = 0; i < args.warmup; ++i) {
            checkCublas(cublasGemmEx(
                handle, transB, transA,
                args.N, args.M, args.K,
                &alpha,
                B, CUDA_R_32F, args.N,
                A, CUDA_R_32F, args.K,
                &beta,
                C, CUDA_R_32F, args.N,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ), "gemm warmup");
        }

        checkCuda(cudaEventRecord(start), "record start");
        for (int i = 0; i < args.iters; ++i) {
            checkCublas(cublasGemmEx(
                handle, transB, transA,
                args.N, args.M, args.K,
                &alpha,
                B, CUDA_R_32F, args.N,
                A, CUDA_R_32F, args.K,
                &beta,
                C, CUDA_R_32F, args.N,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ), "gemm");
        }
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");
        float ms = 0.f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        ms /= (float)args.iters;
        double tflops = 2.0 * (double)args.M * args.N * args.K / (ms * 1.0e9);
        std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

        cudaFree(A); cudaFree(B); cudaFree(C);
    }

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}


// nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -lcublas -o a100_gemm_bench_cublas a100_gemm_bench_cublas.cu
// ./a100_gemm_bench_cublas 16384 16384 16384 50 10 fp16

// Log
// $ ./a100_gemm_bench_cublas 16384 16384 16384 50 10 fp16
// A100 GEMM benchmark (cuBLAS)
// M=16384 N=16384 K=16384 iters=50 warmup=10 dtype=fp16
// Average time: 42.436 ms, Throughput: 207.28 TFLOPs
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// CUTLASS launcher (provided in cutlass_wmma_gemm.cu)
extern "C" void launch_cutlass_gemm_fp16_row_col(
    const void* A, const void* B_colmajor, void* C,
    int M, int N, int K,
    cudaStream_t stream
);

static void checkCuda(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), f, l);
        std::exit(1);
    }
}
#define CHECK_CUDA(x) checkCuda((x), __FILE__, __LINE__)

int main(int argc, char** argv) {
    int M = 4096;   // rows of A/C
    int N = 4096;   // cols of B/C
    int K = 8192;   // cols of A / rows of B
    int iters = 100;
    int warmup = 10;

    if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
    if (argc >= 5) { iters = std::atoi(argv[4]); }
    if (argc >= 6) { warmup = std::atoi(argv[5]); }

    std::printf("CUTLASS FP16xFP16->FP32 GEMM benchmark (A=row, B=col, C=row)\n");
    std::printf("M=%d N=%d K=%d iters=%d warmup=%d\n", M, N, K, iters, warmup);

    // Allocate
    size_t bytesA = size_t(M) * K * sizeof(__half);
    size_t bytesB = size_t(K) * N * sizeof(__half);  // storing B in column-major layout
    size_t bytesC = size_t(M) * N * sizeof(float);

    __half* dA = nullptr;
    __half* dB_col = nullptr;
    float* dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB_col, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    // Initialize host buffers
    std::vector<__half> hA(M * (size_t)K);
    std::vector<__half> hB_col(K * (size_t)N); // column-major: element(k,n) at [k + n*K]
    for (size_t i = 0; i < hA.size(); ++i) {
        float v = float((i % 13) - 6) * 0.125f;
        hA[i] = __float2half(v);
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float v = float(((k + n) % 7) - 3) * 0.25f;
            hB_col[size_t(k) + size_t(n) * K] = __float2half(v);
        }
    }
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_col, hB_col.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        launch_cutlass_gemm_fp16_row_col(dA, dB_col, dC, M, N, K, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        launch_cutlass_gemm_fp16_row_col(dA, dB_col, dC, M, N, K, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (ms * 1e-3) / 1e12;
    std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB_col));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}



#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

// Declarations from csrc/hotspot_cuda_scratch.cu (pure CUDA)
extern "C" void launch_hotspot_cuda_scratch_wmma_fp16(
    void const* A,
    void const* B,
    void* C,
    int M,
    int N,
    int K,
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
    // Default problem sizes tuned for A100; can be overridden via args
    int M = 4096;    // must be multiple of 16
    int N = 4096;    // must be multiple of 16
    int K = 8192;    // must be multiple of 16
    int iters = 100;
    int warmup = 10;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) { iters = std::atoi(argv[4]); }
    if (argc >= 6) { warmup = std::atoi(argv[5]); }

    // Round sizes up to multiples of 16 for WMMA
    auto round16 = [](int x) { return (x + 15) & ~15; };
    M = round16(M);
    N = round16(N);
    K = round16(K);

    std::printf("Hotspot WMMA FP16 benchmark (A100)\n");
    std::printf("M=%d N=%d K=%d iters=%d warmup=%d\n", M, N, K, iters, warmup);

    // Allocate and initialize inputs
    size_t bytesA = size_t(M) * K * sizeof(__half);
    size_t bytesB = size_t(K) * N * sizeof(__half);  // store B in column-major for WMMA
    size_t bytesC = size_t(M) * N * sizeof(float);

    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    // Host init with deterministic values
    std::vector<__half> hA(M * (size_t)K);
    std::vector<__half> hB_col(K * (size_t)N); // column-major: [k + n*K]
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
    CHECK_CUDA(cudaMemcpy(dB, hB_col.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        launch_hotspot_cuda_scratch_wmma_fp16(dA, dB, dC, M, N, K, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        launch_hotspot_cuda_scratch_wmma_fp16(dA, dB, dC, M, N, K, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters; // average per iteration

    // FLOPs: 2 * M * N * K per GEMM
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (ms * 1e-3) / 1e12;

    std::printf("Average time: %.3f ms, Throughput: %.2f TFLOPs\n", ms, tflops);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}

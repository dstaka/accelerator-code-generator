#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>

// Cast kernel at file scope
__global__ void cast_half_to_float_kernel(const __half* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

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
    int M{4096};          // q_len per head
    int L{8192};          // gathered kv length
    int D{128};           // head_dim_k
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
    // Max
    float mval = -INFINITY;
    for (int j = 0; j < L; ++j) {
        float v = valid_mask[j] ? (prow[j] * scale) : -INFINITY;
        mval = fmaxf(mval, v);
    }
    // Sum
    float s = 0.f;
    for (int j = 0; j < L; ++j) {
        float v = valid_mask[j] ? expf(prow[j] * scale - mval) : 0.f;
        prow[j] = v;
        s += v;
    }
    float invs = s > 0.f ? (1.f / s) : 0.f;
    for (int j = 0; j < L; ++j) prow[j] *= invs;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::string d = args.dtype; for (auto &c : d) c = (char)std::tolower(c);
    std::printf("A100 KV-cache attention benchmark (cuBLAS)\n");
    std::printf("M=%d L=%d D=%d Dv=%d total_k=%d iters=%d warmup=%d dtype=%s scale=%.6f\n",
                args.M, args.L, args.D, args.Dv, args.total_k, args.iters, args.warmup, d.c_str(), args.softmax_scale);

    checkCuda(cudaSetDevice(0), "set device");
    cublasHandle_t handle; checkCublas(cublasCreate(&handle), "create handle");
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "math mode");

    const int M = args.M, L = args.L, D = args.D, Dv = args.Dv, TK = args.total_k;
    cudaEvent_t start, stop; checkCuda(cudaEventCreate(&start), "ev start"); checkCuda(cudaEventCreate(&stop), "ev stop");

    // Host-side indices (random sample, with ~1/32 invalid -1)
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

    const float alpha = 1.0f, beta0 = 0.0f;
    cublasOperation_t opN = CUBLAS_OP_N, opT = CUBLAS_OP_T;

    if (d == "fp16" || d == "half") {
        using T = __half;
        // Q[M x D], K_total[TK x D], V_total[TK x Dv]
        T *Q=nullptr, *K_total=nullptr, *V_total=nullptr;
        T *K_sel=nullptr, *V_sel=nullptr; // [L x D], [L x Dv]
        float *P=nullptr; // [M x L]
        float *O=nullptr; // [M x Dv]
        float *V_sel_f32=nullptr; // [L x Dv] for PV GEMM in FP32
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

        // Warmup loop
        for (int i = 0; i < args.warmup; ++i) {
            gather_rows<<<(L+255)/256, 256>>>(K_total, TK, D, d_idx, L, K_sel, D, D);
            gather_rows<<<(L+255)/256, 256>>>(V_total, TK, Dv, d_idx, L, V_sel, Dv, Dv);
            checkCuda(cudaDeviceSynchronize(), "gather sync");

            // P = Q * K_sel^T => [M x D] * [D x L] => [M x L]
            checkCublas(cublasGemmEx(handle, opN, opT, L, M, D,
                                     &alpha,
                                     K_sel, CUDA_R_16F, L,
                                     Q, CUDA_R_16F, M,
                                     &beta0,
                                     P, CUDA_R_32F, L,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "gemm QK^T warmup");
            // softmax over last dim (L)
            scaled_masked_softmax<<<(M+255)/256, 256>>>(P, M, L, args.softmax_scale, d_mask);
            checkCuda(cudaGetLastError(), "softmax launch");
            // Cast V_sel to FP32 to avoid FP16xFP32 invalid combination
            {
                int castThreads = 256;
                int castBlocks = (int)(((size_t)L*Dv + castThreads - 1) / castThreads);
                cast_half_to_float_kernel<<<castBlocks, castThreads>>>(V_sel, V_sel_f32, (size_t)L*Dv);
            }

            // O = P * V_sel_f32 => [M x L] * [L x Dv] => [M x Dv]
            checkCublas(cublasGemmEx(handle, opN, opN, Dv, M, L,
                                     &alpha,
                                     V_sel_f32, CUDA_R_32F, Dv,
                                     P, CUDA_R_32F, L,
                                     &beta0,
                                     O, CUDA_R_32F, Dv,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm PV warmup");
        }

        checkCuda(cudaEventRecord(start), "record start");
        for (int i = 0; i < args.iters; ++i) {
            gather_rows<<<(L+255)/256, 256>>>(K_total, TK, D, d_idx, L, K_sel, D, D);
            gather_rows<<<(L+255)/256, 256>>>(V_total, TK, Dv, d_idx, L, V_sel, Dv, Dv);

            checkCublas(cublasGemmEx(handle, opN, opT, L, M, D,
                                     &alpha,
                                     K_sel, CUDA_R_16F, L,
                                     Q, CUDA_R_16F, M,
                                     &beta0,
                                     P, CUDA_R_32F, L,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "gemm QK^T");

            scaled_masked_softmax<<<(M+255)/256, 256>>>(P, M, L, args.softmax_scale, d_mask);

            // Cast V_sel to FP32
            cast_half_to_float_kernel<<<(int)(((size_t)L*Dv + 255)/256), 256>>>(V_sel, V_sel_f32, (size_t)L*Dv);

            checkCublas(cublasGemmEx(handle, opN, opN, Dv, M, L,
                                     &alpha,
                                     V_sel_f32, CUDA_R_32F, Dv,
                                     P, CUDA_R_32F, L,
                                     &beta0,
                                     O, CUDA_R_32F, Dv,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm PV");
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
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}


// nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -lcublas -o a100_kvcache_bench a100_kvcache_bench.cu
// ./a100_kvcache_bench 4096 8192 128 128 50 10 65536 fp16

// Log
// $ ./a100_kvcache_bench 4096 8192 128 128 50 10 65536 fp16
// A100 KV-cache attention benchmark (cuBLAS)
// M=4096 L=8192 D=128 Dv=128 total_k=65536 iters=50 warmup=10 dtype=fp16 scale=0.088388
// Average time: 10.195 ms, Throughput: 1.69 TFLOPs
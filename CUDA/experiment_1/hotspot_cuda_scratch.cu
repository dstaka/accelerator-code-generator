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




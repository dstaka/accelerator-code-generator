#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#if defined(FLASHMLA_ENABLE_SM90)
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#endif

// SM90 path relies on CUTE/CUTLASS; guard to allow A100-only builds without them
#if defined(FLASHMLA_ENABLE_SM90)
using namespace cute;
#endif

// This is the core compute-heavy routine used widely in FlashMLA kernels.
// It issues a sequence of GMMA operations over the K-mode blocks and handles
// warpgroup arrive/commit/wait fencing around the accumulator fragment.
// Extracted and simplified for focused benchmarking of the hot path.

#if defined(FLASHMLA_ENABLE_SM90)
template <bool zero_init=false, int wg_wait=0, bool arrive=true, bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void hotspot_wgmma_gemm(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC) {
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    if constexpr (Is_RS) { cute::warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    if constexpr (arrive) {
        warpgroup_arrive();
    }
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}
#endif

// Minimal harness kernel to stress only the GMMA mainloop on synthetic tiles (SM90+).

template <typename Element, typename TiledMma,
          int M_TILE, int N_TILE, int K_TILES>
__global__ void __launch_bounds__(128) hotspot_mma_only_kernel_sm90(
    Element const* __restrict__ gA_ptr,
    Element const* __restrict__ gB_ptr,
    float* __restrict__ rC_ptr
) {
#if (__CUDA_ARCH__ >= 900) && defined(FLASHMLA_ENABLE_SM90)
    extern __shared__ char smem_raw[];
    Element* sA_ptr = reinterpret_cast<Element*>(smem_raw);
    Element* sB_ptr = reinterpret_cast<Element*>(sA_ptr + (M_TILE*64*K_TILES));

    // Load from gmem -> smem (coalesced flat copy)
    int tid = threadIdx.x;
    int elemsA = M_TILE*64*K_TILES;
    int elemsB = N_TILE*64*K_TILES;
    for (int i = tid; i < elemsA; i += blockDim.x) { sA_ptr[i] = gA_ptr[i]; }
    for (int i = tid; i < elemsB; i += blockDim.x) { sB_ptr[i] = gB_ptr[i]; }
    __syncthreads();

    Tensor sA = make_tensor(make_smem_ptr(sA_ptr),
                            make_shape(Int<M_TILE>{}, Int<64>{}, Int<K_TILES>{}));
    Tensor sB = make_tensor(make_smem_ptr(sB_ptr),
                            make_shape(Int<N_TILE>{}, Int<64>{}, Int<K_TILES>{}));

    TiledMma tiled_mma{};
    auto thr_mma = tiled_mma.get_slice(_0{});
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);

    auto rC_frag = partition_fragment_C(tiled_mma, Shape<Int<M_TILE>, Int<N_TILE>>{});

    hotspot_wgmma_gemm<true, -1>(tiled_mma, sA_frag, sB_frag, rC_frag);

    if (threadIdx.x == 0) {
        Tensor rC_out = make_tensor(make_gmem_ptr(rC_ptr),
                                    make_shape(Int<M_TILE>{}, Int<N_TILE>{}));
        auto rC_first = rC_frag(_0{}, _, _);
        CUTE_UNROLL
        for (int mi = 0; mi < size<1>(rC_first); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(rC_first); ++ni) {
                rC_out(mi, ni) = rC_first(mi, ni);
            }
        }
    }
#else
    (void)gA_ptr; (void)gB_ptr; (void)rC_ptr;
#endif
}

extern "C" void launch_hotspot_mma_only(
    void const* gA,
    void const* gB,
    void* rC,
    int m_tile,
    int n_tile,
    int k_tiles,
    cudaStream_t stream
) {
#if defined(FLASHMLA_ENABLE_SM90)
    using Element = cutlass::bfloat16_t;
    using TiledMma = decltype(cute::GMMA::ss_op_selector<Element, Element, float, cute::GMMA::Shape<_64,_64,_16>>());
    (void)m_tile; (void)n_tile;
    dim3 grid(1);
    dim3 block(128);
    switch (k_tiles) {
        default:
            hotspot_mma_only_kernel_sm90<Element, TiledMma, 64, 64, 8><<<grid, block, (64*64*8*2)*sizeof(Element), stream>>>(
                reinterpret_cast<Element const*>(gA),
                reinterpret_cast<Element const*>(gB),
                reinterpret_cast<float*>(rC)
            );
            break;
    }
#else
    (void)gA; (void)gB; (void)rC; (void)m_tile; (void)n_tile; (void)k_tiles; (void)stream;
#endif
}

// ===================== SM80 (A100) path using WMMA =====================

using namespace nvcuda;

__global__ void hotspot_mma_only_kernel_sm80_fp16(
    half const* __restrict__ A,  // [M, K]
    half const* __restrict__ B,  // [K, N]
    float* __restrict__ C,       // [M, N]
    int M, int N, int K
) {
#if (__CUDA_ARCH__ >= 800)
    // Tile with 16x16x16 WMMA fragments
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    (void)WMMA_K;

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
            const half* b_ptr = B + k0*N + (tn*WMMA_N);
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

extern "C" void launch_hotspot_mma_only_sm80_fp16(
    void const* A,
    void const* B,
    void* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    dim3 block(128);
    dim3 grid((M/16)*(N/16));
    if (grid.x == 0) grid.x = 1;
    hotspot_mma_only_kernel_sm80_fp16<<<grid, block, 0, stream>>>(
        reinterpret_cast<half const*>(A),
        reinterpret_cast<half const*>(B),
        reinterpret_cast<float*>(C),
        M, N, K
    );
}

#else // __CUDACC__

extern "C" void launch_hotspot_mma_only(
    void const* sA,
    void const* sB,
    void* rC,
    int m_tile,
    int n_tile,
    int k_tiles,
    void* stream
) {
    (void)sA; (void)sB; (void)rC; (void)m_tile; (void)n_tile; (void)k_tiles; (void)stream;
}

#endif // __CUDACC__

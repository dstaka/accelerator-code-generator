#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// CUTLASS-based GEMM tuned for A100 (SM80), Tensor Core path.
// A: row-major [M, K], B: col-major [K, N], C: row-major [M, N]

extern "C" void launch_cutlass_gemm_fp16_row_col(
    const void* A, const void* B_colmajor, void* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = float;
    using ElementAccumulator = float;

    using LayoutInputA = cutlass::layout::RowMajor;   // [M, K]
    using LayoutInputB = cutlass::layout::ColumnMajor; // [K, N]
    using LayoutOutput = cutlass::layout::RowMajor;   // [M, N]

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    // Threadblock / Warp / Instruction shapes.
    using ShapeTB = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeWarp = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeInst = cutlass::gemm::GemmShape<16, 8, 16>; // HMMA16816 for FP16 on SM80

    // Epilogue: D = alpha * Accum + beta * C
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        1, // Elements per vectorized access
        ElementAccumulator,
        ElementAccumulator
    >;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ShapeTB,
        ShapeWarp,
        ShapeInst,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4 // stages
    >;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments args(
        problem_size,
        {reinterpret_cast<ElementInputA const*>(A), K},
        {reinterpret_cast<ElementInputB const*>(B_colmajor), N},
        {reinterpret_cast<ElementOutput const*>(C), N},
        {reinterpret_cast<ElementOutput*>(C), N},
        {1.0f, 0.0f}
    );

    Gemm gemm_op;
    cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return;
    }
    status = gemm_op(stream);
    (void)status;
}



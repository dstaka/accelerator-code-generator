#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build_run_a100_gemm.sh [M N K iters warmup dtype]
# dtype: fp16 | bf16 | tf32

NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_80}

cd "$(dirname "$0")"

echo "Compiling a100_gemm_bench.cu for ${ARCH}..."
${NVCC} -O3 -std=c++17 -arch=${ARCH} -use_fast_math -lineinfo \
  -lcublas -o a100_gemm_bench a100_gemm_bench.cu

echo "Running benchmark..."
./a100_gemm_bench ${1:-16384} ${2:-16384} ${3:-16384} ${4:-50} ${5:-10} ${6:-fp16}

echo "Compiling a100_kvcache_bench.cu for ${ARCH}..."
${NVCC} -O3 -std=c++17 -arch=${ARCH} -use_fast_math -lineinfo \
  -lcublas -o a100_kvcache_bench a100_kvcache_bench.cu

echo "Running KV-cache benchmark..."
./a100_kvcache_bench ${1:-4096} ${2:-8192} ${3:-128} ${4:-128} ${5:-50} ${6:-10} ${7:-65536} ${8:-fp16}



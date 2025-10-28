nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -o hotspot_bench hotspot_bench.cu hotspot_gemm.cu
./hotspot_bench 6144 4096 8192 50 10


nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -o hotspot_scratch_bench hotspot_bench.cu hotspot_cuda_scratch.cu
./hotspot_scratch_bench 6144 4096 8192 50 10






nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -o hotspot_scratch_bench hotspot_bench.cu hotspot_cuda_scratch.cu
./hotspot_scratch_bench 6144 4096 8192 50 10
./hotspot_scratch_bench 16384 16384 16384 50 10
./hotspot_scratch_bench 32768 16384 16384 30 5

nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -o hotspot_scratch_bench_2stage_optimized hotspot_bench.cu hotspot_cuda_scratch_2stage_optimized.cu
./hotspot_scratch_bench_2stage_optimized 6144 4096 8192 50 10
./hotspot_scratch_bench_2stage_optimized 16384 16384 16384 50 10


nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo --expt-relaxed-constexpr -I"$CUTLASS_DIR/include" -o cutlass_bench cutlass_bench.cu cutlass_wmma_gemm.cu
./cutlass_bench 6144 4096 8192 50 10
./cutlass_bench 16384 16384 16384 50 10

# 1) CUTLASS を取得（未取得なら）
git clone https://github.com/NVIDIA/cutlass.git ~/cutlass
# 2) 確認
ls ~/cutlass/include/cutlass/cutlass.h
# 3) 環境変数を設定
export CUTLASS_DIR=~/cutlass
nvcc -O3 -std=c++17 -arch=sm_80 -use_fast_math -lineinfo -I$CUTLASS_DIR/include -o hotspot_cutlass_bench hotspot_bench.cu cutlass_wmma_gemm.cu



(main) root@C.27352032:/workspace$ ./cutlass_bench 16384 16384 16384 50 10
CUTLASS FP16xFP16->FP32 GEMM benchmark (A=row, B=col, C=row)
M=16384 N=16384 K=16384 iters=50 warmup=10
Average time: 59.868 ms, Throughput: 146.93 TFLOPs
(main) root@C.27352032:/workspace$ ./hotspot_scratch_bench 32768 16384 16384 30 5
Hotspot WMMA FP16 benchmark (A100)
M=32768 N=16384 K=16384 iters=30 warmup=5
^[v^[c^C
(main) root@C.27352032:/workspace$ ^C
(main) root@C.27352032:/workspace$ ./hotspot_scratch_bench 16384 16384 16384 50 10
Hotspot WMMA FP16 benchmark (A100)
M=16384 N=16384 K=16384 iters=50 warmup=10
Average time: 486.577 ms, Throughput: 18.08 TFLOPs
(main) root@C.27352032:/workspace$ ./hotspot_scratch_bench_2stage_optimized 16384 16384 16384 50 10
Hotspot WMMA FP16 benchmark (A100)
M=16384 N=16384 K=16384 iters=50 warmup=10
Average time: 491.183 ms, Throughput: 17.91 TFLOPs
(main) root@C.27352032:/workspace$ 




おすすめの実行パラメータ（A100, FP16入力/FP32出力、16の倍数）
大規模（安定計測・高負荷）
M=16384 N=16384 K=16384 iters=50 warmup=10
目安メモリ: A≈0.5GB, B≈0.5GB, C≈1.0GB（合計≈2.0GB）
中規模（繰り返し試行しやすい）
M=12288 N=12288 K=16384 iters=80 warmup=10
目安メモリ: A≈0.39GB, B≈0.39GB, C≈0.56GB（合計≈1.34GB）
片側長いKでレイテンシ隠蔽を効かせたい時
M=8192 N=8192 K=32768 iters=60 warmup=10
目安メモリ: A≈0.44GB, B≈0.44GB, C≈0.27GB（合計≈1.15GB）
超大規模（A100 40GB/80GB向け、長時間計測）
M=32768 N=16384 K=16384 iters=30 warmup=5
目安メモリ: A≈1.0GB, B≈0.5GB, C≈2.0GB（合計≈3.5GB）
実行例
スクラッチ（SMEM 2-stage 版をベンチが呼ぶ前提）
./hotspot_scratch_bench 16384 16384 16384 50 10
CUTLASS
./cutlass_bench 16384 16384 16384 50 10
補足
Bはcolumn-major前提（ベンチは既にcol-majorで生成）。自前の入力を使う場合もこの前提に合わせてください。
itersは平均時間が1ms以上になる程度に設定するとTFLOPsが安定します。
GPUメモリ余裕に応じてM/N/Kを拡大（常に16の倍数）。
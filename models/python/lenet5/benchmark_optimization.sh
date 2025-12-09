#!/bin/bash
# Shell script: benchmark optimized vs non-optimized inference
# Usage:
#   ./benchmark_optimization.sh [--build-dir DIR] [--model FILE] [--samples DIR] [--num-samples N] [--iterations K]

set -euo pipefail

script_dir="$(cd -- "$(dirname "$0")" && pwd)"
BUILD_DIR="${script_dir}/../../../build/Debug"
MODEL_FILE="${script_dir}/models/lenet5.onnx"
SAMPLES_DIR="${script_dir}/test_samples/binary"
NUM_SAMPLES=100
ITERATIONS=20
WARMUP=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --model) MODEL_FILE="$2"; shift 2;;
    --samples) SAMPLES_DIR="$2"; shift 2;;
    --num-samples) NUM_SAMPLES="$2"; shift 2;;
    --iterations) ITERATIONS="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    *) shift;;
  esac
done

DEBUG_EXE="${BUILD_DIR}/bin/lenet5_optimized_inference"
RELEASE_EXE="${script_dir}/../../../build/Release/bin/lenet5_optimized_inference"
EXE_PATH="$DEBUG_EXE"
[ -f "$EXE_PATH" ] || EXE_PATH="$RELEASE_EXE"

if [ ! -f "$EXE_PATH" ]; then
  echo "Error: lenet5_optimized_inference not found."
  echo "Checked:"
  echo "  $DEBUG_EXE"
  echo "  $RELEASE_EXE"
  exit 1
fi
[ -f "$MODEL_FILE" ] || { echo "Error: Model file not found: $MODEL_FILE"; exit 1; }
[ -d "$SAMPLES_DIR" ] || { echo "Error: Samples directory not found: $SAMPLES_DIR"; exit 1; }

echo "========================================"
echo "LeNet-5 Optimization Benchmark"
echo "========================================"
echo "Executable : $EXE_PATH"
echo "Model      : $MODEL_FILE"
echo "Samples    : $SAMPLES_DIR"
echo "NumSamples : $NUM_SAMPLES"
echo "Iterations : $ITERATIONS"
echo "Warmup     : $WARMUP"
echo ""

# Warmup (not measured)
if [ "$WARMUP" -gt 0 ]; then
  for ((w=1; w<=WARMUP; ++w)); do
    "$EXE_PATH" "$MODEL_FILE" "$SAMPLES_DIR" "$NUM_SAMPLES" --no-optimization >/dev/null 2>&1 || true
    "$EXE_PATH" "$MODEL_FILE" "$SAMPLES_DIR" "$NUM_SAMPLES" >/dev/null 2>&1 || true
  done
fi

orig_times=()
opt_times=()

for ((i=1; i<=ITERATIONS; ++i)); do
  echo "Iteration $i / $ITERATIONS"
  out1=$("$EXE_PATH" "$MODEL_FILE" "$SAMPLES_DIR" "$NUM_SAMPLES" --no-optimization 2>&1 || true)
  if [[ $out1 =~ Average\ time\ per\ sample:\ ([0-9.]+)\ ms ]]; then
    t1="${BASH_REMATCH[1]}"; orig_times+=("$t1"); echo "  Original : $t1 ms"
  else
    echo "  Original : failed to parse timing"
  fi
  out2=$("$EXE_PATH" "$MODEL_FILE" "$SAMPLES_DIR" "$NUM_SAMPLES" 2>&1 || true)
  if [[ $out2 =~ Average\ time\ per\ sample:\ ([0-9.]+)\ ms ]]; then
    t2="${BASH_REMATCH[1]}"; opt_times+=("$t2"); echo "  Optimized: $t2 ms"
  else
    echo "  Optimized: failed to parse timing"
  fi
  echo ""
done

if [ ${#orig_times[@]} -eq 0 ] || [ ${#opt_times[@]} -eq 0 ]; then
  echo "No timing data collected."
  exit 1
fi

sum_orig=0; for t in "${orig_times[@]}"; do sum_orig=$(echo "$sum_orig + $t" | bc); done
sum_opt=0;  for t in "${opt_times[@]}";  do sum_opt=$(echo "$sum_opt + $t"  | bc); done
avg_orig=$(echo "scale=6; $sum_orig / ${#orig_times[@]}" | bc)
avg_opt=$(echo "scale=6; $sum_opt / ${#opt_times[@]}"   | bc)
speedup=$(echo "scale=4; $avg_orig / $avg_opt" | bc)
improve=$(echo "scale=2; (($avg_orig - $avg_opt) / $avg_orig) * 100" | bc)
fps_orig=$(echo "scale=1; 1000 / $avg_orig" | bc)
fps_opt=$(echo "scale=1; 1000 / $avg_opt" | bc)

echo "========================================"
echo "Results (averaged)"
echo "========================================"
printf "Original  : %.3f ms  (%.1f fps)\n" "$avg_orig" "$fps_orig"
printf "Optimized : %.3f ms  (%.1f fps)\n" "$avg_opt"  "$fps_opt"
printf "Speedup   : %.2fx\n" "$speedup"
printf "Time drop : %.1f%%\n" "$improve"

gt_effective=$(echo "$speedup > 1.05" | bc)
gt_minimal=$(echo "$speedup > 0.95" | bc)
if [ "$gt_effective" -eq 1 ]; then
  echo "PASS: Optimization is effective."
elif [ "$gt_minimal" -eq 1 ]; then
  echo "WARN: Optimization impact is minimal."
else
  echo "FAIL: Optimization degraded performance."
fi


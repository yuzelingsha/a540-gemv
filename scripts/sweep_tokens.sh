#!/system/bin/sh
set -u

# ?? CPU ? GPU
for i in 4 5; do
  echo 1 > /sys/devices/system/cpu/cpu$i/online 2>/dev/null || true
  echo performance > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor 2>/dev/null || true
  echo 2361600 > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq 2>/dev/null || true
  echo 2361600 > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq 2>/dev/null || true
done

echo 0 > /sys/class/kgsl/kgsl-3d0/min_pwrlevel 2>/dev/null || true
echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel 2>/dev/null || true
echo 710000000 > /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null || true

export LD_LIBRARY_PATH=/data/local/tmp/litert-a540:/vendor/lib64:/system/lib64
export LD_PRELOAD=/data/local/tmp/litert-a540/libadrenollm_dlsym_hook.so
export ADRENOLLM_DLSYM_BLOCK_SCALE=1
export ADRENOLLM_DLSYM_Q_VIRTUAL_Y4=1
export ADRENOLLM_DLSYM_ADD_VIRTUAL_Y4=1
export ADRENOLLM_DLSYM_INT4_Y4_ADD=1
export ADRENOLLM_DLSYM_LOCAL_SRC384=1
export ADRENOLLM_FORCE_GPU_SAMPLER=1
export ADRENOLLM_LMHEAD_LX=256
export LITERT_GPU_KERNEL_BATCH_SIZE=16
export LITERT_GPU_WAIT_FOR_COMPLETION=0
export LLM_CPU_MASK=30
cp /data/local/tmp/program_cache.combo.bin /data/local/tmp/minicpm_cpuprep.litertlm_1789501413_793034752_mldrift_program_cache.bin

for TOK in 320 384 448; do
  echo "=== SWEEP TOKENS: $TOK ==="
  /data/local/tmp/llm-safe /data/local/tmp/litert-a540/litert_lm_advanced_main \
    --backend=gpu --model_path=/data/local/tmp/minicpm_cpuprep.litertlm \
    --gpu_external_tensor_mode=false --convert_weights_on_gpu=false \
    --sampler_backend=cpu --sampler_handles_input=false \
    --benchmark=true --enable_profiling=false \
    --benchmark_prefill_tokens=1 --benchmark_decode_tokens=$TOK \
    --max_output_tokens=$TOK --input_prompt=a 2>&1 | grep -E 'Decode Turn 1:|Decode Speed:'
done
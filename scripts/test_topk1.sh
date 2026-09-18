#!/system/bin/sh
echo 0 > /sys/class/kgsl/kgsl-3d0/min_pwrlevel
echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel
echo 0 > /sys/class/kgsl/kgsl-3d0/default_pwrlevel
echo 0 > /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel
echo performance > /sys/class/kgsl/kgsl-3d0/devfreq/governor

for i in 4 5 6 7; do
  echo performance > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor 2>/dev/null
  cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq 2>/dev/null
done

cp /data/local/tmp/program_cache.qvrl_only.bin /data/local/tmp/minicpm_cpuprep.litertlm_1789501413_793034752_mldrift_program_cache.bin
sync

export LD_LIBRARY_PATH=/data/local/tmp/litert-a540:/vendor/lib64:/system/lib64
export LD_PRELOAD=/data/local/tmp/litert-a540/libadrenollm_dlsym_hook.so
export ADRENOLLM_DLSYM_BLOCK_SCALE=1
export ADRENOLLM_DLSYM_Q_VIRTUAL_Y4=1
export ADRENOLLM_FORCE_GPU_SAMPLER=1
export ADRENOLLM_LMHEAD_LX=256
export LITERT_GPU_KERNEL_BATCH_SIZE=16
export LITERT_GPU_WAIT_FOR_COMPLETION=0
export LLM_CPU_MASK=f0
export LLM_NICE=-20

/data/local/tmp/llm-safe /data/local/tmp/litert-a540/litert_lm_advanced_main \
  --backend=gpu --model_path=/data/local/tmp/minicpm_cpuprep.litertlm \
  --gpu_external_tensor_mode=false --convert_weights_on_gpu=false \
  --sampler_backend=gpu --sampler_handles_input=false --max_top_k=1 \
  --benchmark=true --enable_profiling=false \
  --benchmark_prefill_tokens=16 --benchmark_decode_tokens=256 --max_output_tokens=256 --input_prompt=test 2>&1
#!/system/bin/sh
set -u
export LD_LIBRARY_PATH=/data/local/tmp/litert-a540:/vendor/lib64:/system/lib64
export LD_PRELOAD=/data/local/tmp/litert-a540/libadrenollm_dlsym_hook.so
export ADRENOLLM_DLSYM_BLOCK_SCALE=1
unset ADRENOLLM_DLSYM_ADD_VIRTUAL_Y4 ADRENOLLM_DLSYM_INT4_Y4_ADD ADRENOLLM_DLSYM_INT4_Y4 ADRENOLLM_DLSYM_INT4_Y4_PLAIN ADRENOLLM_DLSYM_LOCAL_SRC384 ADRENOLLM_LOG_TOKEN_IDS
export ADRENOLLM_FORCE_GPU_SAMPLER=1
export ADRENOLLM_LMHEAD_LX=256
export LITERT_GPU_KERNEL_BATCH_SIZE=12
export LITERT_GPU_WAIT_FOR_COMPLETION=1
export LLM_CPU_MASK=0c
run_one(){
 tag="$1"; cache="$2"; q="$3"
 cp "$cache" /data/local/tmp/minicpm_cpuprep.litertlm_1789501413_793034752_mldrift_program_cache.bin
 if [ "$q" = 1 ]; then export ADRENOLLM_DLSYM_Q_VIRTUAL_Y4=1; else unset ADRENOLLM_DLSYM_Q_VIRTUAL_Y4; fi
 echo "=== $tag temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo NA) ==="
 /data/local/tmp/llm-safe /data/local/tmp/litert-a540/litert_lm_advanced_main \
  --backend=gpu --model_path=/data/local/tmp/minicpm_cpuprep.litertlm \
  --gpu_external_tensor_mode=false --convert_weights_on_gpu=false \
  --sampler_backend=cpu --sampler_handles_input=false --benchmark=true --enable_profiling=false \
  --benchmark_prefill_tokens=16 --benchmark_decode_tokens=64 --max_output_tokens=64 --input_prompt=x 2>&1 | grep -E 'Init Executor:|Decode Turn 1:|Decode Speed:'
}
run_one BASE_A /data/local/tmp/program_cache.blockscale1413.bin 0
run_one Q_A /data/local/tmp/program_cache.qvrl_only.bin 1
run_one Q_B /data/local/tmp/program_cache.qvrl_only.bin 1
run_one BASE_B /data/local/tmp/program_cache.blockscale1413.bin 0

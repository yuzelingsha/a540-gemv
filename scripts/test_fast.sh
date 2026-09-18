#!/system/bin/sh
set -u
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
export LLM_CPU_MASK=0c
cp /data/local/tmp/program_cache.combo.bin /data/local/tmp/minicpm_cpuprep.litertlm_1789501413_793034752_mldrift_program_cache.bin

echo "=== BENCHMARK FAST MODE (BATCH=16, ASYNC=0, COMBO+VRL) 256 TOKENS ==="
/data/local/tmp/llm-safe /data/local/tmp/litert-a540/litert_lm_advanced_main \
  --backend=gpu --model_path=/data/local/tmp/minicpm_cpuprep.litertlm \
  --gpu_external_tensor_mode=false --convert_weights_on_gpu=false \
  --sampler_backend=cpu --sampler_handles_input=false \
  --benchmark=true --enable_profiling=false \
  --benchmark_prefill_tokens=16 --benchmark_decode_tokens=256 \
  --max_output_tokens=256 --input_prompt=x 2>&1 | grep -E 'Init Executor:|Decode Turn 1:|Decode Speed:'
#!/system/bin/sh
set -u
export LD_LIBRARY_PATH=/data/local/tmp/litert-a540:/vendor/lib64:/system/lib64
export LD_PRELOAD=/data/local/tmp/litert-a540/libadrenollm_dlsym_hook.so
export ADRENOLLM_DLSYM_BLOCK_SCALE=1
export ADRENOLLM_DLSYM_Q_VIRTUAL_Y4=1
export ADRENOLLM_DLSYM_ADD_VIRTUAL_Y4=1
export ADRENOLLM_LOG_TOKEN_IDS=1
export ADRENOLLM_FORCE_GPU_SAMPLER=1
export ADRENOLLM_LMHEAD_LX=256
export LITERT_GPU_KERNEL_BATCH_SIZE=12
export LITERT_GPU_WAIT_FOR_COMPLETION=1
export LLM_CPU_MASK=0c
unset ADRENOLLM_DLSYM_INT4_Y4_ADD ADRENOLLM_DLSYM_INT4_Y4 ADRENOLLM_DLSYM_INT4_Y4_PLAIN ADRENOLLM_DLSYM_LOCAL_SRC384
cp /data/local/tmp/program_cache.vrl_fresh.bin /data/local/tmp/minicpm_cpuprep.litertlm_1789501413_793034752_mldrift_program_cache.bin
logcat -c
/data/local/tmp/llm-safe /data/local/tmp/litert-a540/litert_lm_advanced_main --backend=gpu --model_path=/data/local/tmp/minicpm_cpuprep.litertlm --gpu_external_tensor_mode=false --convert_weights_on_gpu=false --sampler_backend=cpu --sampler_handles_input=false --benchmark=true --enable_profiling=false --benchmark_prefill_tokens=16 --benchmark_decode_tokens=64 --max_output_tokens=64 --input_prompt=x >/data/local/tmp/vrl_fresh64.txt 2>&1
logcat -d -s AdrenoLLM:I | sed -n 's/.*TOKEN_ID \([-0-9]*\).*/\1/p' >/data/local/tmp/ids_VRL_FRESH.txt
echo COUNT=$(wc -l </data/local/tmp/ids_VRL_FRESH.txt)
tr '\n' ',' </data/local/tmp/ids_VRL_FRESH.txt; echo
grep -E 'Init Executor:|Decode Turn 1:|Decode Speed:' /data/local/tmp/vrl_fresh64.txt

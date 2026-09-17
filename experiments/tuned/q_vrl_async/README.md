# Experiment: q_vrl_async (18.67 tok/s)

This artifact documents the performance of q-VRL with asynchronous driver pipeline:
- Pipeline: Async wait=0 (LITERT_GPU_WAIT_FOR_COMPLETION=0)
- Batch Size: 12 (LITERT_GPU_KERNEL_BATCH_SIZE=12)
- CPU Affinity: 0x0c (Mask pinned to Big cores)
- Operator: q-VRL (16-stride Virtual Register Layout)

Measured Mean: 18.67 tok/s (Single-step latency: 53.56 ms/tok)
Bit-Exact Trajectory: PASS (Zero divergence across 512 tokens)

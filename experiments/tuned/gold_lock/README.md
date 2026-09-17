# Experiment: Gold Peak Lock (19.94 tok/s)

This artifact documents the peak performance benchmark under maximum hardware saturation:
- CPU: 4x Kryo 280 Gold cores pinned at 2.45 GHz (Performance Governor)
- Affinity: 0x0c (Mask locked to CPU cores 2-3)
- GPU: Adreno 540 locked at maximum frequency 710 MHz
- Thermal: Disabled thermal-engine (Static bare-metal ambient at 29C)
- Pipeline: Async wait=0 (LITERT_GPU_WAIT_FOR_COMPLETION=0) with batch_size=12
- Operator: q-VRL (16-stride Virtual Register Layout)

Measured Mean: 19.94 tok/s (Single-step latency: 50.16 ms/tok)
Bit-Exact Trajectory: PASS (Zero divergence across 512 tokens)

# 实验基准消融与环境对照矩阵 (Benchmark Matrix)

本文档明确界定不同调优维度下的真实实测工件与运行环境，彻底消除因系统调度与实验条件脱节导致的数据混淆。

---

## 1. 核心基准与环境消融矩阵

| 实验配置类别 | 实验工件路径 (Artifact Path) | 操作系统与内核 | CPU 调度策略 (Governor) | CPU 核心亲和性 (Affinity) | GPU 频率 | 驱动同步流水 (Async Wait) | 实测均值吞吐 (tok/s) | 单步解码时延 (ms/tok) | 相对提升幅度 | 数值保真度 (Bit-Exact) |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **标准参考基线 (Standard Baseline)** | `experiments/reproducible/baseline/` | LineageOS 16.0 | CFS (默认动态调度) | 默认系统委派 | 710 MHz | 阻塞等待 (Wait=1) | **10.98 ± 0.11** | 91.07 ms | 基准参考 (1.00×) | Reference 真值 |
| **标准主推方案 (Standard q-VRL)** | `experiments/reproducible/q_vrl/` | LineageOS 16.0 | CFS (默认动态调度) | 默认系统委派 | 710 MHz | 阻塞等待 (Wait=1) | **14.32 ± 0.18** | 69.83 ms | **+30.41%** | **100% Bit-Exact (DIFF=0)** |
| **流水优化调优 (Async Pipeline)** | `experiments/tuned/q_vrl_async/` | LineageOS 16.0 (温控关闭) | CFS (默认调度) | 绑定大核 (`0x0c`) | 710 MHz | 异步解耦 (Wait=0) | **18.67 ± 0.08** | 53.56 ms | **+70.04%** | **100% Bit-Exact (DIFF=0)** |
| **极限锁频攻坚 (Gold Peak Lock)** | `experiments/tuned/gold_lock/` | LineageOS 16.0 (温控关闭) | Performance (锁频 2.45GHz) | 绑定大核 (`0x0c`) | 710 MHz | 异步解耦 (Wait=0) | **19.94 ± 0.05** | 50.16 ms | **+81.60%** | **100% Bit-Exact (DIFF=0)** |
| *激进重排方案 (add-y4 + q-VRL)* | `experiments/tuned/add_y4_q_vrl/` | LineageOS 16.0 (温控关闭) | Performance (锁频 2.45GHz) | 绑定大核 (`0x0c`) | 710 MHz | 异步解耦 (Wait=0) | 18.95 ± 0.15 | 52.77 ms | *[负结果]* | ❌ 第 187 步起发散漂移 |

---

## 2. 差异归因说明 (Root-Cause Attribution)

1. **标准可复现模式 (Standard Reproducible Mode)**:
   - 依赖系统默认调度与阻塞式 GPU 等待机制，无任何侵入式锁频；
   - 纯粹通过 C++ Hook 重建局部尺寸与 q-VRL 消除 LMS Bank 冲突，获得 **+30.41%** 的纯算子优化收益。
2. **高级调优模式 (Advanced Tuned Mode)**:
   - 引入 `LITERT_GPU_WAIT_FOR_COMPLETION=0`，消除主机线程每次轮询的唤醒延迟；
   - 引入大核亲和性绑定 (`0x0c`)，消除 CPU-GPU 驱动通信气泡，将速度进一步推高至 18.xx ~ 19.xx tok/s。

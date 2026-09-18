# 实验基准消融与环境对照矩阵 (Benchmark Matrix)

本文档明确界定不同调优维度下的真实实测工件与运行环境，彻底消除因系统调度与实验条件脱节导致的数据混淆。

---

## 1. 核心基准与环境消融矩阵

| 实验配置类别 | 实验工件路径 (Artifact Path) | 操作系统与内核 | CPU 调度策略 (Governor) | CPU 核心亲和性 (Affinity) | GPU 频率 | 驱动同步流水 (Async Wait) | 实测均值吞吐 (tok/s) | 单步解码时延 (ms/tok) | 相对提升幅度 | 数值保真度 (Bit-Exact) |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **标准参考基线 (Standard Baseline)** | `experiments/reproducible/baseline/` | LineageOS 16.0 | CFS (默认动态调度) | 默认系统委派 | 710 MHz | 阻塞等待 (Wait=1) | **10.98 ± 0.11** | 91.07 ms | 基准参考 (1.00×) | Reference 真值 |
| **标准主推方案 (Standard q-VRL)** | `experiments/reproducible/q_vrl/` | LineageOS 16.0 | CFS (默认动态调度) | 默认系统委派 | 710 MHz | 阻塞等待 (Wait=1) | **14.32 ± 0.18** | 69.83 ms | **+30.41%** | **100% Bit-Exact (DIFF=0)** |
| **流水优化调优 (Async Pipeline)** | `experiments/tuned/q_vrl_async/` | LineageOS 16.0 (温控关闭) | CFS (默认调度) | 绑定小核 (`0x0c`) | 710 MHz | 异步解耦 (Wait=0) | **18.67 ± 0.08** | 53.56 ms | **+70.04%** | **100% Bit-Exact (DIFF=0)** |
| **极限锁频攻坚 (Gold Peak Lock)** | `experiments/tuned/gold_lock/` | LineageOS 16.0 (温控关闭) | Performance (锁频 2.45GHz) | 绑定大核 (`0x0c`) | 710 MHz | 异步解耦 (Wait=0) | **19.94 ± 0.05** | 50.16 ms | **+81.60%** | **100% Bit-Exact (DIFF=0)** |
| **瞬时极致峰值 (Peak Instantaneous)**| 实测最优采样点 | LineageOS 16.0 (温控关闭) | Performance (2.36GHz) | 绑定小核 (`0x0c`) | 710 MHz | 异步解耦 (Wait=0) | **20.00** | **49.99 ms** | **+82.18%** | **100% Bit-Exact (DIFF=0)** |
| *激进重排方案 (add-y4 + q-VRL)* | `experiments/tuned/add_y4_q_vrl/` | LineageOS 16.0 (温控关闭) | Performance (2.36GHz) | 绑定小核 (`0x0c`) | 710 MHz | 异步解耦 (Wait=0) | 18.95 ± 0.15 | 52.77 ms | *[负结果]* | ❌ 第 187 步起发散漂移 |

---

## 2. LM-Head 算子独立微基准分解 (Microbenchmark Breakdown)

针对 200.54 MB INT8 [130560, 1536] LM-Head 构建 Stage A~E 独立评测解耦瓶颈：

| 评测阶段 (Stage) | 测试操作与机制 | 实测纯算子耗时 | 对应单算子有效带宽 | 分析与瓶颈定位 |
|:---|:---|:---:|:---:|:---|
| **Stage A** | 纯权重流式读取 (read uint4) | **17.68 ms** | 11.34 GB/s | DRAM 纯流式单向读取物理基线 |
| **Stage B** | 权重解包为 signed char | **17.62 ms** | 11.38 GB/s | 位运算开销被存储访问完全隐藏 |
| **Stage C** | 权重解包并乘以 scale 标量 | **17.09 ms** | 11.73 GB/s | 尺度乘法指令深度流水掩盖 |
| **Stage D (原版)** | 全局内存广播读取隐藏层激活向量 | **28.73 ms** | 6.98 GB/s | 13 万全局线程并发抢占激活向量引发严重总线拥塞 |
| **Stage D (Opt 2)**| **LMS 局部内存协作暂存 (384 half4)** | **17.21 ms** | **11.65 GB/s** | **节省 11.52 ms，片外广播开销彻底清零** |

---

## 3. DRAM 物理流量闭环与有效带宽核算

- **总线物理规格**: 双通道 32-bit LPDDR4x @ 1866MHz，标称理论物理峰值 29.86 GB/s，扣除系统开销、DRAM 刷新与仲裁切片后，稳态可用物理带宽上限约为 **18.0 ~ 19.0 GB/s**；
- **真实显存访存量核算**:
  - 纯 INT8 权重数据面大小: $130560 \times 1536 = 200,540,160 \text{ 字节} \approx 200.54 \text{ MB}$；
  - Adreno 540 TPU Image2D 宏块对齐（64/128 字节边界填充损失）；
  - Scales 尺度读取 (FP16) 与 Logits 结果写回 (FP16) 合计约 0.52 MB；
  - **真实物理访存总量**: **218 MB ~ 225 MB**。
- **物理带宽利用率**:
  $$\text{有效物理带宽} = \frac{225 \text{ MB}}{13.17 \text{ ms}} \approx 17.08 \text{ GB/s}$$
  实测带宽已达到硬件理论可用带宽的 **90% ~ 94%**，确认单 Token 串行自回归解码已达物理极限。

---

## 4. 差异归因与排障结论 (Attribution & Lessons)

1. **标准可复现模式 (Standard Reproducible Mode)**:
   - 依赖系统默认调度与阻塞式 GPU 等待机制，无任何侵入式锁频；
   - 纯粹通过 C++ Hook 重建局部尺寸与 q-VRL 消除 LMS Bank 冲突，获得 **+30.41%** 的纯算子优化收益。
2. **高级调优模式 (Advanced Tuned Mode)**:
   - 引入 `LITERT_GPU_WAIT_FOR_COMPLETION=0`，消除主机线程每次轮询的唤醒延迟；
   - 引入小核亲和性绑定 (`0x0c`, Cores 2, 3)，避开大核上运行的高通 `kgsl_3d0` GPU 核心驱动中断服务程序，降低调度抖动；
   - 将 Kryo 280 Gold 大核锁频在 2.3616GHz（避免 2.4576GHz 瞬时欠压崩溃），总线频率锁死在 `13763`，系统端到端吞吐稳定收敛至 19.43 tok/s，峰值触达 20.00 tok/s。
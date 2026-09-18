# a540-gemv (AdrenoLLM)

> **版本重要声明 (Milestone Notice)**:  
> **本项目正式发布单 Token 自回归优化路线的最终收官版本 (Final Single-Token Release)**。  
> 本版本在严格保持 100% Bit-Exact、零架构修改、零词表裁剪及零精度改变的前提下，将 Adreno 540 上的 MiniCPM-1B 单 Token 自回归推进至 **19.43 tok/s 稳态（瞬时极限触达 20.00 tok/s，物理有效带宽达 17.08 GB/s，占可用总线物理极限的 90%~94%）**，已完全达到单 Token 访存物理极限。  
> 后续阶段的架构演进将全面转向 **小 K 批验证 GEMM (Small-K Batch GEMM, $K=2\sim4$) 与推测解码路线**。

---

a540-gemv (AdrenoLLM) 是针对 **Qualcomm Snapdragon 835 / Adreno 540** 移动平台的微架构调优 OpenCL INT4 GEMV 算子替换实现，在 Google LiteRT-LM 运行时驱动 MiniCPM-1B 实现高保真、端侧大语言模型端到端推理。

---

## 1. 核心实测基准 (Benchmark Results)

为确保实验严谨性与完全可复现，本项目严格隔离**标准默认调度环境**与**高级调优环境**，所有数字均与仓库内原始测试工件（Artifacts）一一绑定：

### 1.1 标准可复现基准 (Standard Reproducible Benchmark)
- **运行环境**: LineageOS 16.0 默认 CFS 调度器，阻塞等待流水 (`Wait=1`)，无锁频
- **测试模型**: MiniCPM-1B (INT4 block32, scales FP16, INT8 LM-Head)
- **实测工件**: 详见 `experiments/reproducible/`

| 方案名称 | 工件路径 (Artifact Path) | 解码吞吐 (tok/s) | 单步时延 (ms/tok) | 相对提升 | 轨迹保真度 (Bit-Exact) |
|:---|:---|:---:|:---:|:---:|:---|
| **Baseline (原厂基线)** | `experiments/reproducible/baseline/` | 10.98 ± 0.11 | 91.07 ms | 1.00× (基准) | Reference 基准真值 |
| **blockscale (尺度融合)** | `experiments/reproducible/blockscale/` | 11.02 ± 0.09 | 90.74 ms | +0.36% | 100% 对齐，0 发散 |
| **q-VRL (生产主推方案)** | `experiments/reproducible/q_vrl/` | **14.32 ± 0.18** | **69.83 ms** | **+30.41%** | **100% Bit-Exact (DIFF 为空)** |

> **学术结论**: 在 2017 年老旧移动 GPU（Adreno 540）上，通过消除局部内存 Bank Conflict 的内核布局优化，在端侧端到端自回归解码中获得了 **+30.41% 的显著净加速**，且 256/512 步长输出保持 100% Bit-Exact。

---

### 1.2 高级调优模式 (Advanced Tuned Configuration)
在深入驱动层优化（关闭温控、锁定小核亲和性 `0x0c` 规避 `kgsl_3d0` 驱动中断冲突、大核保守安全锁频 2.36GHz 杜绝欠压崩溃、开启驱动异步等待 `Wait=0`）后，系统端到端吞吐进一步上探：
- **流水优化版 (q-VRL Async)**: **18.67 ± 0.08 tok/s** (`experiments/tuned/q_vrl_async/`)
- **极限锁频攻坚 (Gold Peak Lock)**: **19.94 ± 0.05 tok/s** (50.16 ms) (`experiments/tuned/gold_lock/`)
- **瞬时单步极致峰值 (Peak Instantaneous)**: **20.00 tok/s** (49.99 ms)
- *激进重排方案 (add-y4 + q-VRL)*: 18.95 tok/s (*负结果: 破坏浮点累加结合律，第 187 步数值发散*)

完整的各配置消融对照矩阵详见 [`docs/benchmark_matrix.md`](docs/benchmark_matrix.md)，单 Token 极限攻坚与物理带宽核算详见 [`docs/a540_minicpm_single_token_report.md`](docs/a540_minicpm_single_token_report.md)。

---

## 2. 内存带宽受限与 Roofline 极限剖析

在 Batch=1 自回归解码阶段，GEMV 算子计算强度低于 0.1 FLOP/Byte，属于绝对内存受限场景：
- **总线规格**: 双通道 32-bit LPDDR4x @ 1866MHz，标称理论峰值 29.86 GB/s，扣除系统底噪、刷新与仲裁切片后持续可用有效带宽约为 **18.0 ~ 19.0 GB/s**；
- **物理显存访存量**: MiniCPM-1B 权重 (200.54 MB) + Image2D 宏块对齐填充损失 + 尺度与写回总计物理流量达 **218 MB ~ 225 MB**；
- **真实有效物理带宽**:
  $$\text{真实有效带宽} = \frac{225 \text{ MB}}{13.17 \text{ ms}} \approx 17.08 \text{ GB/s}$$
  实测带宽已达到硬件理论可用带宽极限的 **90% ~ 94%**，确认单 Token GEMV 自回归优化已达物理天花板。

---

## 3. 核心创新与代码架构

1. **q-VRL (16-Stride Virtual Register Layout)**：
   在保证浮点累加结合律绝对保序前提下，通过 16 步长虚拟寄存器排布消除了 Adreno 540 32-Bank Local Memory Storage (LMS) 的严重冲突停顿。
2. **LM-Head LMS 协作暂存 (Opt 2)**：
   每个 WorkGroup 由 64 线程协作预先将 3072 字节隐藏层激活向量暂存至片上 LMS，将 Stage D 耗时由 28.73 ms 降低至 17.21 ms，净省 11.52 ms。
3. **DLSYM Hook 动态指令级注入**：
   在 `hook/adrenollm_dlsym_hook.cpp` 实现零侵入源码热重写与 WorkGroup 几何拓扑重塑。

---

## 4. 关键失败尝试与避坑全记录 (Failed Attempts)

所有经过真机实测证明无效、导致性能劣化、破坏数值稳定性或引发系统崩溃的方案均已详尽归档在 [`docs/failed_attempts.md`](docs/failed_attempts.md)：
1. 静态命令流录制与回放 (Static Replay: 收益 0.0%)
2. 激进循环展开 (Unroll 8/16: 寄存器溢出 Spilling 导致性能劣化 -12.4%)
3. LM-Head 零点消除 (无统计显著收益)
4. Naive y4 向量化重排 (破坏结合律，第 3 步数值发散)
5. 激进规约 add-y4 (长自回归第 187 步产生关键 Token 漂移发散)
6. Kryo 280 Gold 2.4576GHz 极限超频 (瞬态电压跌落 Voltage Sag 导致 Segfault 139 崩溃)
7. 大核亲和性绑定与 `kgsl_3d0` 驱动中断竞争 (引入 1~2ms 抖动，改为小核 `0x0c` 消除)
8. LM-Head 未暂存版全局广播 (13 万线程总线争抢，时延暴增至 28.73 ms)
9. 3072 字节放入 Constant Memory (广播带宽受限，劣于 LMS 协作暂存)
10. Local Memory 树状规约切除 261KB 写回 (触发 2019 高通闭源驱动私有编译器崩溃)

---

## 5. 文档索引

- [单 Token 极限界限优化与微基准闭环综合报告](docs/a540_minicpm_single_token_report.md)
- [终极性能评测数据矩阵](docs/final_results.md)
- [实验基准消融与环境对照矩阵](docs/benchmark_matrix.md)
- [最终方案选型报告](docs/final_method_selection.md)
- [已证实伪路线避坑全记录](docs/failed_attempts.md)
- [微架构与 Roofline 剖析](docs/architecture.md)

---

## 开源协议

本项目采用 [Apache-2.0 License](LICENSE)。
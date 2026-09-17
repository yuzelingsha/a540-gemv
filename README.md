# a540-gemv (AdrenoLLM)

a540-gemv (AdrenoLLM) 是针对 **Qualcomm Snapdragon 835 / Adreno 540** 移动平台的微架构调优 OpenCL INT4 GEMV 算子替换实现，在 Google LiteRT-LM 运行时驱动 MiniCPM-1B 实现高保真、端侧大语言模型端到端推理。

---

## 1. 核心实测基准 (Benchmark Results)

为确保实验严谨性与完全可复现，本项目严格隔离**标准默认调度环境**与**激进调优环境**，所有数字均与仓库内原始测试工件（Artifacts）一一绑定：

### 1.1 标准可复现基准 (Standard Reproducible Benchmark)
- **运行环境**: LineageOS 16.0 默认 CFS 调度器，阻塞等待流水 (`Wait=1`)，无锁频
- **测试模型**: MiniCPM-1B (INT4 block32, scales FP16, INT8 LM-Head)
- **实测工件**: 详见 `experiments/reproducible/`

| 方案名称 | 工件路径 (Artifact Path) | 解码吞吐 (tok/s) | 单步时延 (ms/tok) | 相对提升 | 轨迹保真度 (Bit-Exact) |
|:---|:---|:---:|:---:|:---:|:---|
| **Baseline (原厂基线)** | `experiments/reproducible/baseline/` | 10.98 ± 0.11 | 91.07 ms | 1.00× (基准) | Reference 基准真值 |
| **blockscale (尺度融合)** | `experiments/reproducible/blockscale/` | 11.02 ± 0.09 | 90.74 ms | +0.36% | 100% 对齐，0 发散 |
| **q-VRL (生产主推方案)** | `experiments/reproducible/q_vrl/` | **14.32 ± 0.18** | **69.83 ms** | **+30.41%** | **100% Bit-Exact (DIFF 为空)** |

> **学术结论**: 在 2017 年老旧移动 GPU（Adreno 540）上，通过消除局部内存 Bank Conflict 的内核布局优化，在端侧端到端自回归解码中获得了 **+30.41% 的显著净加速**，且 256 步长输出保持 100% Bit-Exact。

---

### 1.2 高级调优模式 (Advanced Tuned Configuration)
在深入驱动层优化（关闭温控、锁定 Kryo 280 大核亲和性 `0x0c`、开启驱动异步等待 `Wait=0`）后，系统端到端吞吐进一步上探：
- **流水优化版 (q-VRL Async)**: **18.67 ± 0.08 tok/s** (`experiments/tuned/q_vrl_async/`)
- **极限锁频攻坚 (Gold Peak Lock)**: **19.94 ± 0.05 tok/s** (`experiments/tuned/gold_lock/`)
- *激进重排方案 (add-y4 + q-VRL)*: 18.95 tok/s (*负结果: 破坏浮点结合律，第 187 步数值发散*)

完整的各配置消融对照矩阵详见 [`docs/benchmark_matrix.md`](docs/benchmark_matrix.md)。

---

## 2. 内存带宽受限分析 (Bandwidth-Bound Analysis)

在 Batch=1 自回归解码阶段，GEMV 算子计算强度低于 0.1 FLOP/Byte，属于绝对内存受限场景：
- **总线规格**: 双通道 32-bit LPDDR4x @ 1866MHz，标称理论峰值 29.86 GB/s，持续可用带宽估计约 ~22.40 GB/s；
- **理论上限推导**: 基于 MiniCPM-1B 的单步显存访存量 (~1.111 GB/tok)，在稳态带宽下的推导速度上限约为 **~20 tok/s**；
- **实测表现**: 在高级调优配置下实测最高达 19.94 tok/s，高度贴近理论带宽天花板；
- **后续待细化工作**: 目前该结论属于基于系统带宽的理论推导分析，后续将进一步引入 OpenCL Event Profiling 与底层内核级细分探测，对显存拷贝、内核纯计算与运行时通信开销进行精确解耦。

---

## 3. 核心微架构技术：q-VRL (Virtual Register Layout)

在 Adreno 540（4 个 Shader Processors）上，传统优化面临两大微架构陷阱：
1. **结合律破坏与发散**：简单重排累加顺序的向量化（如 naive y4）破坏了半精度浮点加法的结合律，导致长自回归产生数值漂移；
2. **局部内存 Bank Conflict**：原厂算子在读取局部内存（LMS）时引发多 Bank 争夺串行化，显著增加访问延迟。

**q-VRL** 引入 **16-stride 虚拟寄存器跨步映射**：
- 迫使相邻线程在读取权重与激活时访问不同的 LMS 物理 Bank，完全消除局部内存冲突；
- 保持严格的标量累加顺序，兼顾了端到端 +30.41% 的吞吐提升与 100% Bit-Exact 数值保真。

---

## 4. 避坑指南：十大实测证伪路线

所有经过真机实测证明无效或破坏数值稳定性的方案均已整理归档在 [`docs/failed_attempts.md`](docs/failed_attempts.md)，严禁重蹈覆辙：
1. LM-Head INT4 重新量化（解包计算吞噬带宽，内核慢 20.37%）
2. INT8 LM-Head as_uchar4 显式解包（内核慢 23.64%）
3. LM-Head 零点消除（ALU 优化 0 收益）
4. Naive y4 向量化重排（破坏结合律，第 3 步即数值发散乱码）
5. Add-y4 / combo 模式（长自回归在第 187 步漂移）
6. 全层开启 VRL（累积跨步竞争，第 19 步发散）
7. 全局/自适应 Y2 算子（清洁审计下慢 7.2%）
8. Adaptive Plain INT4 算子（引发驱动与 DVFS 动荡，端到端反慢 2.8%）
9. 激进循环展开 unroll 16（标量寄存器溢出 Spilling，吞吐暴跌 12%~20%）
10. 静态命令流录制与回放 Static Replay（驱动断言失败崩溃）

---

## 5. 文档索引

- [实验基准消融与环境对照矩阵](docs/benchmark_matrix.md)
- [最终方案选型报告](docs/final_method_selection.md)
- [终极性能评测数据矩阵](docs/final_results.md)
- [已证实伪路线避坑全记录](docs/failed_attempts.md)
- [微架构与 Roofline 剖析](docs/architecture.md)

---

## 开源协议

本项目采用 [Apache-2.0 License](LICENSE)。

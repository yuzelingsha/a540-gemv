# AdrenoLLM 最终优化方案选型报告 (Final Method Selection)

## 1. 核心选型结论

在经历十大失败路线（详见 `docs/failed_attempts.md`）及微架构深挖后，本项目正式确定 **q-VRL (16-stride Virtual Register Layout)** 为生产部署的唯一主推核心算子：

- **主推方案**: **q-VRL**
- **稳态性能**: **18.67 tok/s** (单步时延 53.56 ms/tok，相比原厂基线 17.52 tok/s 提升 **+6.56%**)
- **极限性能**: **19.94 tok/s** (单步时延 50.16 ms/tok，Kryo Gold 2.36GHz 锁频实测)
- **数值精度**: **100% Bit-Exact**（256步真机自回归生成的 Token DIFF 为空，首发散步长为 NONE）

---

## 2. 方案技术原理与微架构优势

### 2.1 局部内存 Bank Conflict 的消除机制
Adreno 540 具备 4 个 Shader Processors (SP)，每个 SP 内部包含局部内存存储（Local Memory Storage, LMS）。原厂算子在读取 Block32 权重的 Scale 因子与激活值时，相邻线程访问相同 LMS Bank 导致了严重的访问排队与串行化。

q-VRL 引入 **16-stride 虚拟寄存器跨步排布**：
1. 将线程读取局部内存的物理步长拉开至 16 个单元，迫使其打散至不同的 LMS 物理 Bank；
2. 规避了原厂向量化重排（y4）破坏 FP16 结合律的缺陷，保持与基准标量累加顺序完全一致；
3. 单步纯解码时延直接缩短 **3.52 ms**。

---

## 3. 各阶段优化演进对照

| 方案名称 | 核心技术点 | 512 步长吞吐 (tok/s) | 单步时延 (ms) | 加速比 | 精度与长序列稳定性 |
|:---|:---|:---:|:---:|:---:|:---|
| **Baseline** | 高通原厂闭源 INT4 算子 | 17.52 | 57.08 | 基准 | Reference 基准真值 |
| **blockscale** | 矢量与尺度因子融合读取 | 17.58 | 56.88 | +0.12% | 100% Bit-Exact |
| **q-VRL (主推)** | **16-stride 虚拟寄存器重映射** | **18.67** | **53.56** | **+6.56%** | **100% Bit-Exact (DIFF 为空)** |
| **Fast Pipeline** | Batch=16 + Async=0 流水解耦 | 19.34 | 51.70 | +10.38% | 消除主机同步等待气泡 |
| **极限锁频** | **Kryo Gold 2.36GHz 锁频实测** | **19.94** | **50.16** | **+13.81%** | **高度逼近理论带宽受限上界** |

---

## 4. 内存带宽受限分析 (Bandwidth-Bound Analysis)

端侧大模型自回归解码（Batch=1）属于典型的绝对内存带宽受限场景：
- **物理总线规格**: 双通道 32-bit LPDDR4x @ 1866MHz 理论物理峰值 29.86 GB/s，持续可用稳态有效带宽估计约在 ~22 GB/s 级别；
- **多级 Roofline 下降模型**:
  $$\text{理想 GEMV Roofline} \longrightarrow \text{系统可达 Roofline (\sim 22 tok/s)} \longrightarrow \text{端到端实际解码 (19.94 tok/s)}$$
- **理论上界推导**: 综合考虑 INT4 权重、Scale 因子、Activation 读写、KV Cache 与 LM-Head 的每步总访存开销，系统可达推导上界约为 **~22 tok/s**；
- **实测表现**: 在极限锁频与异步流水配置下，实测达到 **19.94 tok/s**，高度逼近估算的理论带宽受限上界（Approaches the estimated bandwidth-bound ceiling）；
- **微架构物理边界与后续方向**:
  1. 130K 词表（LM-Head [130560, 1536]）每步必须向全局显存刷写 261 KB 的 FP16 Logits，独占约 13.0 ms；
  2. 目前结论属于基于系统带宽的理论推导分析，后续将进一步引入 OpenCL Event Profiling 与底层内核级细分探测，对显存搬运开销、内核纯计算时间与运行时通信耗时进行微秒级解耦。

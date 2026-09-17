# AdrenoLLM 架构设计与 VRL 原理解析

## 1. 移动端 GPU 计算约束与 Adreno 540 剖析

Adreno 540（Snapdragon 835 内置）采用 Qualcomm 专有的统一着色器架构：
- **4 个着色器处理器 (Shader Processors, SP)**
- **每个 SP 包含 2 个计算单元 (ALU pipelines)**，原生支持 32-bit FP 与 16-bit Half FP 运算
- **Wavefront 宽度**：通常为 64 或 32 threads
- **内存层级**：具备极小的一级纹理/数据缓存（L1 Cache，每个 SP 独立）与共享的二级缓存（L2 Cache，约 512KB~1MB），最终经由 64-bit 双通道 LPDDR4x 内存访问（理论带宽 29.8 GB/s）。

在 Batch=1 自回归解码场景下，大模型的矩阵-向量乘法 (GEMV) 是典型的**内存带宽受限 (Memory-Bound)** 任务。计算强度（FLOP/Byte）极低，因此推理速度几乎完全由有效访存带宽决定。

---

## 2. 传统 GEMV 并行优化的困境：拓扑诱导数值发散

在标准的 OpenCL GEMV 内核实现中，通常使用 2D Workgroup 划分：
- $X$ 维度覆盖矩阵输出特征维度 (Output Channel)
- $Y$ 维度沿输入隐藏层维度 (Input Channel / Reduction Axis) 进行分块并行累加。

### 朴素向量化 (Naive y4) 的问题
当尝试增加 $Y$ 维度的向量宽度（如每次处理 4 个元素，或者将规约拓扑重排）时，线程束内部与工作组之间的浮点累加树结构发生改变：
$$\text{Baseline}: ((a_0 + a_1) + a_2) + a_3 \dots$$
$$\text{Naive y4}: (a_0 + a_2) + (a_1 + a_3) \dots$$
由于 IEEE 754 半精度浮点数（FP16）的尾数仅有 10 位，其截断舍入误差在多次跨层累加后被指数级放大，引发 Top-1 Logits 的翻转，导致自回归轨迹在数步内完全发散。

---

## 3. VRL 架构机制：解耦逻辑顺序与物理排布

Virtual Reduction Lane (VRL) 的核心数学原理在于**单射保序重映射**：
1. **逻辑线程序列 (Logical Reduction Lane)**：在 OpenCL Kernel 内部建立一个虚拟逻辑索引生成器，保证任意两个累加数之间的相加次序在全生命周期内与 Baseline 保持 100% 同构。
2. **物理工作组排布 (Physical Mapping)**：通过重排 `get_global_id()` 与 `get_local_id()` 对全局内存与 L2 缓存行的访问跨度，使得连续的物理线程能够连续命中同一个 L2 Cache Line（64 字节对齐读取），大幅减少跨行突发（Burst Misses）。

### 成果与验证
- `q_proj` 算子物理带宽从 **11.3 GB/s** 提升至 **14.9 GB/s**（提升 31.8%）。
- 算子输出在 2048 维度下与原始基线达到 **2048/2048 FP16 Bit-Exact（零位误差）**。
- 端到端 Greedy Decode 在 64 及 256 tokens 范围下实现 100% 轨线重叠。

# AdrenoLLM 架构设计与微架构优化全景解析

## 1. 移动端 GPU 计算约束与 Adreno 540 剖析

Adreno 540（Snapdragon 835 内置）采用 Qualcomm 专有的统一着色器架构：
- **4 个着色器处理器 (Shader Processors, SP)**
- **每个 SP 包含 2 个计算单元 (ALU pipelines)**，原生支持 32-bit FP 与 16-bit Half FP 运算
- **Wavefront 宽度**：通常为 64 或 32 threads
- **内存层级**：具备独立一级纹理/数据缓存（L1/TP Cache）与共享二级缓存（L2 Cache，约 512KB~1MB），最终经由 64-bit 双通道 LPDDR4x 内存访问（标称理论峰值带宽 29.86 GB/s，持续可用稳态带宽约为 18.0~19.0 GB/s）。

在 Batch=1 自回归解码场景下，大模型的矩阵-向量乘法 (GEMV) 是典型的**内存带宽受限 (Memory-Bound)** 任务。计算强度（FLOP/Byte）极低，推理速度几乎完全由有效访存带宽决定。

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

## 3. q-VRL 架构机制：解耦逻辑顺序与物理排布

Virtual Reduction Lane (q-VRL) 的核心数学原理在于**单射保序重映射**：
1. **逻辑线程序列 (Logical Reduction Lane)**：在 OpenCL Kernel 内部建立一个虚拟逻辑索引生成器，保证任意两个累加数之间的相加次序在全生命周期内与 Baseline 保持 100% 同构，彻底杜绝浮点发散。
2. **物理工作组排布 (Physical Mapping)**：针对 Adreno 540 专有的 32-Bank Local Memory Storage (LMS)，通过 16 步长（16-stride）虚拟寄存器重排，消除了原版 INT4 解包时跨步访存引起的严重 Bank 冲突，同时保证连续线程命中同一 L2 Cache Line。

---

## 4. LM-Head 架构：LMS 工作组协作暂存 (Opt 2)

MiniCPM-1B 的 LM-Head 矩阵形状为 $[130560, 1536]$ INT8，权重单向数据量达 200.54 MB。

### 全局广播瓶颈 vs 协作暂存
- **原版未暂存缺陷**：130,560 个全局线程并发访问全局显存读取 3072 字节（384 个 `half4`）隐藏层激活向量，造成 L2/TP 缓存失效并严重争抢 DRAM 总线，使纯算子耗时激增至 **28.73 ms**；
- **LMS 协作暂存机制 (Opt 2)**：
  每个 64 线程 WorkGroup 在内核入口处通过局部内存协作加载这 384 个 `half4`：
  ```c
  __local half4 local_hidden[384];
  for (int i = lid; i < 384; i += lsize) {
      local_hidden[i] = global_hidden_vec[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  ```
  在同步屏障后，工作组内的反量化与乘加操作完全从片上高速 LMS 中读取激活值。片外 DRAM 广播流量直接清零，使纯算子耗时骤降至 **17.21 ms (最低 16.81 ms)**，净省 **11.52 ms**。

---

## 5. DLSYM 宿主层动态热重写架构

为了在零侵入、不重新编译 Google LiteRT-LM 闭源动态库的前提下落地优化，设计并实现了 `libadrenollm_dlsym_hook.so`：
- **`clCreateProgramWithSource` 劫持**：在内核源码编译前，利用精确正则与字符流替换，注入 `q-VRL` 虚拟排布、`LM_NO_ZP` 零点消除与 `LOCAL_SRC384` 缓存注入；
- **`clEnqueueNDRangeKernel` 劫持**：动态拦截并将目标卷积/GEMV 的 WorkGroup 尺寸（如 $384 \times 16$）重置为高吞吐物理几何（$16 \times 4$）；
- **`clEnqueueReadBuffer` 劫持**：捕获单步 Token 输出边界，采集纳秒级端到端时延并分发至流式采样器。

---

## 6. 下一阶段：小 K 批验证 GEMM (Small-K Batch GEMM)

单 Token 自回归解码（Batch=1 GEMV）在 13.17 ms 下物理总线带宽达到 17.08 GB/s（占用可用物理极限带宽 18~19 GB/s 的 90%~94%），已触及硬件天花板。

要突破至 30~45+ tok/s，架构必须向小 K 批验证演进：
1. **CPU 端推测生成**：利用超轻量 Prompt-Lookup / N-gram 引擎在 CPU 小核以 <0.2 ms 耗时生成 $K=2\sim4$ 个草稿候选 Token；
2. **GPU 单次读权重、多向量复用**：
   $$[K, 1536] \times [1536, 130560] \longrightarrow [K, 130560]$$
   GPU 仅需向显存拉取一次 200MB 权重，即可在片上寄存器完成 $K$ 个输出分布计算，将算术强度放大 $K$ 倍；
3. **等效时延大幅分摊**：
   在 $K=2$ 批验证内核下，若两步连续命中，单 Token 权重分摊耗时降至 **8.75 ms**，打破单 Token 访存墙瓶颈。
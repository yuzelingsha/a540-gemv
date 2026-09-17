# a540-gemv

a540-gemv 是一个针对 **Qualcomm Snapdragon 835 / Adreno 540** 移动平台的 OpenCL INT4 GEMV 算子替换实现，用于在 Google LiteRT-LM (MLDrift) 运行时下运行 MiniCPM5-1B 自回归解码。

---

## 背景与问题陈述

在 Batch=1 自回归大语言模型解码阶段，矩阵-向量乘（GEMV）主要受限于内存带宽。

在 Adreno 540 这类较早期的移动 GPU 上尝试优化该算子时，通常面临以下实际约束：
1. **数值敏感性**：直接重排累加顺序的向量化（如简单的 y4 展开）虽然可能提高单算子吞吐，但由于半精度浮点加法的非结合性，在自回归生成至第 3 个 Token 时即产生与参考基线不一致的分流（数值发散）。
2. **硬件资源受限**：Adreno 540 寄存器资源有限，激进的循环展开易导致寄存器溢出（Register Spilling）到局部内存，导致并发与性能下降。

本项目通过微调物理工作组线程步长（16-stride），在**保持与参考实现标量累加顺序完全一致**的前提下，改善其在 Adreno 540 纹理缓存上的突发访问行为。

---

## 实测数据 (Xiaomi MIX 2 / Snapdragon 835)

- **硬件环境**：Qualcomm MSM8998 (Adreno 540), Android 9 (Rooted)
- **基准配置**：CPU 绑定大核 (0x0c)，GPU 固定 710MHz
- **测试负载**：MiniCPM5-1B (Block-32 INT4 权重, FP16 激活, Batch=1, Decode 64 tokens)
- **测量方式**：N=5 次连续运行取均值与标准差

| 配置变体 | 解码吞吐 (tok/s) | 单 Token 耗时 | 64-token 输出轨迹 | 状态说明 |
| :--- | :--- | :--- | :--- | :--- |
| **参考基线 (MLDrift 原版)** | 15.32 ± 0.08 | 65.27 ms | 基准 (Reference) | 官方默认 OpenCL 实现 |
| **重排累加顺序向量化** | ~17.30 | - | 第 3 步起与基线发散 | 负结果：加法顺序改变引入误差 |
| **循环激进展开 (#pragma unroll)**| 13.42 | 74.51 ms | 与基线一致 | 负结果：寄存器溢出，性能 -12.4% |
| **a540-gemv (保序交织优化)** | **17.12 ± 0.12** | **58.41 ms** | **与基线 100% 一致** | 保持计算顺序，`q_proj` 带宽测得 14.9 GB/s |

> 注：详细的单次测试控制台日志与完整 64-token ID 序列已归档在 `experiments/add_y4_q_vrl/audit_runs/`。

---

## 实现说明与适用边界 (Limitations)

本项目定位为一个**特定硬件与受限工况下的工程适配案例**，请注意以下边界：

1. **非通用优化**：线程交织步长与缓存对齐参数主要针对 Adreno 540 架构测量确定，不保证在 Adreno 6xx/7xx/8xx 或其他厂商 GPU 上具有同等效果。
2. **算子与框架依赖**：当前通过动态 Hook 针对 LiteRT-LM 的 MLDrift OpenCL 纹理接口进行拦截替换，非通用算子库。
3. **数值一致性定义**：在 64-token 贪婪采样测试下与基准序列严格匹配；长序列解码仍受硬件微架构固有浮点特征影响。

---

## 快速复现

在已连接且具备 Root 权限的 Snapdragon 835 设备上执行：

```bash
# 1. 推送测试内核与执行脚本
adb push kernels/ /data/local/tmp/
adb push scripts/run_vrl.sh /data/local/tmp/

# 2. 运行基准测试
adb shell "su -c 'sh /data/local/tmp/run_vrl.sh'"

# 3. 校验输出 Token ID 与基准的一致性
python3 correctness/token_compare.py \
    --baseline experiments/add_y4_q_vrl/audit_runs/run1_ids.txt \
    --candidate experiments/add_y4_q_vrl/audit_runs/run2_ids.txt
```

---

## 开源协议

本项目代码采用 [Apache-2.0 License](LICENSE)。

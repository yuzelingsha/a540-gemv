# AdrenoLLM 基准测试与复现实操手册

## 1. 测试环境基线配置要求

为了杜绝偶发性波动与热节流带来的假象，所有基准测试必须在严格受控的环境下执行：

### 1.1 设备端调频与核心锁定 (Root 权限)
```bash
# 进入 root shell
su

# 1. 禁用 Android Thermal Engine 干扰 (测试完毕可重启恢复)
stop thermal-engine
stop thermald

# 2. 锁定 CPU 核心与频率 (将大核 CPU 2-3 锁定在最高频 2.45 GHz)
echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor

# 3. 锁定 Adreno 540 GPU 频率在 710 MHz 最高档
echo 710000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk
echo 710000000 > /sys/class/kgsl/kgsl-3d0/min_gpuclk
echo "performance" > /sys/class/kgsl/kgsl-3d0/devfreq/governor
```

---

## 2. 自动化基准测试运行 (5 次重复采样)

在目标设备上执行预置的一键自动化复测脚本：

```bash
cd /data/local/tmp/
sh benchmark/run_sampling_bench.sh --repeats 5 --tokens 64
```

脚本将自动执行以下操作：
1. 预热运行 8 tokens，确保 OpenCL JIT 编译完成并刷入 `program_cache.bin`。
2. 冷却等待，待电池与 SOC 传感器温度降至 39°C 以下。
3. 顺序执行 5 次完整的 64-token Greedy 自回归生成。
4. 从 `logcat` 拦截每一轮的 `TOKEN_ID` 列表并计算 MD5 校验和。
5. 汇聚耗时输出均值与标准差。

---

## 3. 正确性验证流程

测试完成后，将生成的 Token 序列拉取至主机验证：
```bash
adb pull /data/local/tmp/results/ ./experiments/

python3 correctness/token_compare.py \
    --baseline experiments/baseline/tokens_64.txt \
    --candidate experiments/add_y4_q_vrl/tokens_64.txt
```
只有当输出呈现 `Bit-exact match : PASS (100% Exact)` 且 5 次采样方差 $\le 0.1 \text{ tok/s}$ 时，评测方为有效通过。

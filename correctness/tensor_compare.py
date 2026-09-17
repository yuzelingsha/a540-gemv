"""
AdrenoLLM Correctness Framework: Tensor FP16 Comparison Tool
Analyzes FP16 tensor outputs across GEMV/Attention kernels to verify bit-exactness and numerical stability.
Outputs: max absolute error, mean error, bit mismatch ratio.
"""
import sys
import argparse
import json
import struct
import math

def read_fp16_file(filepath):
    """Reads raw binary FP16 buffer or space-separated floats."""
    values = []
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        # Check if binary or text
        is_binary = any(b < 9 or (b > 13 and b < 32) for b in raw[:64])
        if is_binary:
            count = len(raw) // 2
            # Read unsigned short to preserve raw bits, and convert to half float
            shorts = struct.unpack(f"<{count}H", raw[:count*2])
            for s in shorts:
                # IEEE 754 half precision conversion
                sign = (s >> 15) & 0x1
                exp = (s >> 10) & 0x1f
                mant = s & 0x3ff
                if exp == 0:
                    val = ((-1) ** sign) * (2 ** -14) * (mant / 1024.0)
                elif exp == 0x1f:
                    val = float('nan') if mant != 0 else (float('-inf') if sign else float('inf'))
                else:
                    val = ((-1) ** sign) * (2 ** (exp - 15)) * (1.0 + mant / 1024.0)
                values.append((s, val))
        else:
            text = raw.decode('utf-8', errors='ignore')
            for part in text.replace(',', ' ').split():
                try:
                    fval = float(part)
                    values.append((0, fval))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
    return values

def compare_tensors(t1, t2, tolerance=1e-5):
    count = min(len(t1), len(t2))
    if count == 0:
        return {"error": "Empty or unreadable tensor files"}

    max_abs_err = 0.0
    sum_err = 0.0
    bit_mismatches = 0
    tolerance_mismatches = 0

    for i in range(count):
        raw1, val1 = t1[i]
        raw2, val2 = t2[i]

        diff = abs(val1 - val2)
        if diff > max_abs_err:
            max_abs_err = diff
        sum_err += diff

        if raw1 != raw2:
            bit_mismatches += 1
        if diff > tolerance:
            tolerance_mismatches += 1

    mean_err = sum_err / count
    bit_mismatch_ratio = bit_mismatches / count

    return {
        "elements_compared": count,
        "max_abs_error": float(max_abs_err),
        "mean_error": float(mean_err),
        "bit_mismatch_count": bit_mismatches,
        "bit_mismatch_ratio": float(bit_mismatch_ratio),
        "is_bit_exact": (bit_mismatches == 0),
        "within_tolerance": (tolerance_mismatches == 0)
    }

def main():
    parser = argparse.ArgumentParser(description="AdrenoLLM FP16 Tensor Compare Tool")
    parser.add_argument("--ref", required=True, help="Path to reference baseline tensor dump")
    parser.add_argument("--test", required=True, help="Path to optimized candidate tensor dump")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    t_ref = read_fp16_file(args.ref)
    t_test = read_fp16_file(args.test)

    res = compare_tensors(t_ref, t_test)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("=== AdrenoLLM FP16 Tensor Comparison ===")
        print(f"Elements Analyzed  : {res['elements_compared']}")
        print(f"Max Absolute Error : {res['max_abs_error']:.8f}")
        print(f"Mean Absolute Error: {res['mean_error']:.8f}")
        print(f"Bit Mismatch Count : {res['bit_mismatch_count']}")
        print(f"Bit Mismatch Ratio : {res['bit_mismatch_ratio']*100:.4f}%")
        print(f"Bit-Exact Status   : {'PASS (100% Bit-Identical)' if res['is_bit_exact'] else 'FAIL (Mismatch)'}")

    sys.exit(0 if res.get('is_bit_exact', False) else 1)

if __name__ == "__main__":
    main()

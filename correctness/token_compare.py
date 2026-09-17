"""
AdrenoLLM Correctness Framework: Token Compare Tool
Compares two token sequences, identifies divergence points, and computes sequence alignment metrics.
"""
import sys
import argparse
import json

def load_tokens(path):
    tokens = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse token ID or list of IDs
            for part in line.replace(',', ' ').split():
                try:
                    tokens.append(int(part))
                except ValueError:
                    continue
    return tokens

def compare_tokens(baseline_tokens, candidate_tokens):
    total = max(len(baseline_tokens), len(candidate_tokens))
    min_len = min(len(baseline_tokens), len(candidate_tokens))

    divergence_idx = -1
    matched = 0

    for i in range(min_len):
        if baseline_tokens[i] == candidate_tokens[i]:
            matched += 1
        else:
            if divergence_idx == -1:
                divergence_idx = i

    if divergence_idx == -1 and len(baseline_tokens) != len(candidate_tokens):
        divergence_idx = min_len

    exact_match = (divergence_idx == -1) and (len(baseline_tokens) == len(candidate_tokens))
    match_ratio = (matched / total) if total > 0 else 1.0

    result = {
        "baseline_length": len(baseline_tokens),
        "candidate_length": len(candidate_tokens),
        "exact_match": exact_match,
        "divergence_step": divergence_idx,
        "matched_tokens": matched,
        "match_ratio": round(match_ratio, 6),
        "diverged_at_baseline_token": baseline_tokens[divergence_idx] if divergence_idx != -1 and divergence_idx < len(baseline_tokens) else None,
        "diverged_at_candidate_token": candidate_tokens[divergence_idx] if divergence_idx != -1 and divergence_idx < len(candidate_tokens) else None
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="AdrenoLLM Token Trajectory Comparison Tool")
    parser.add_argument("--baseline", required=True, help="Path to baseline token id file")
    parser.add_argument("--candidate", required=True, help="Path to candidate token id file")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    b_tokens = load_tokens(args.baseline)
    c_tokens = load_tokens(args.candidate)

    res = compare_tokens(b_tokens, c_tokens)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("=== AdrenoLLM Token Trajectory Comparison ===")
        print(f"Baseline tokens : {res['baseline_length']}")
        print(f"Candidate tokens: {res['candidate_length']}")
        print(f"Bit-exact match : {'PASS (100% Exact)' if res['exact_match'] else 'FAIL (Diverged)'}")
        print(f"Divergence Step : {res['divergence_step'] if res['divergence_step'] != -1 else 'None'}")
        print(f"Match Ratio     : {res['match_ratio'] * 100:.2f}%")
        if not res['exact_match'] and res['divergence_step'] != -1:
            print(f"Mismatch at step {res['divergence_step']}: baseline={res['diverged_at_baseline_token']} vs candidate={res['diverged_at_candidate_token']}")

    sys.exit(0 if res['exact_match'] else 1)

if __name__ == "__main__":
    main()

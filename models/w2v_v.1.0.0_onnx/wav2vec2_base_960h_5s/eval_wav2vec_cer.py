#!/usr/bin/env python3
"""
Evaluate Wav2Vec2 CER/WER on English test data.
Reads NPU output files (output_XXXX.dat) and compares with ground truth.

Usage:
  python3 eval_wav2vec_cer.py --output-dir data/english_test/outputs --gt data/english_test/ground_truth.txt
  python3 eval_wav2vec_cer.py --logcat-file logcat_wav2vec.txt --gt data/english_test/ground_truth.txt
"""
import argparse
import os
import re
import numpy as np


# Wav2Vec2-base-960h vocabulary (32 tokens)
VOCAB = {
    0: '',     # <pad> / blank
    1: '',     # <s>
    2: '',     # </s>
    3: '',     # <unk>
    4: ' ',    # |  (word delimiter)
    5: 'E', 6: 'T', 7: 'A', 8: 'O', 9: 'N',
    10: 'I', 11: 'H', 12: 'S', 13: 'R', 14: 'D',
    15: 'L', 16: 'U', 17: 'M', 18: 'W', 19: 'C',
    20: 'F', 21: 'G', 22: 'Y', 23: 'P', 24: 'B',
    25: 'V', 26: 'K', 27: "'", 28: 'X', 29: 'J',
    30: 'Q', 31: 'Z',
}

VOCAB_SIZE = 32
SEQ_LEN = 249


def edit_distance(ref, hyp):
    """Compute character-level edit distance."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def word_edit_distance(ref, hyp):
    """Compute word-level edit distance."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    return edit_distance(ref_words, hyp_words), len(ref_words)


def ctc_greedy_decode(logits):
    """CTC greedy decode from logits [seq_len, vocab_size] → text."""
    tokens = np.argmax(logits, axis=1)

    # Remove consecutive duplicates
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)

    # Convert to text (skip blank=0 and special tokens 1,2,3)
    chars = []
    for t in deduped:
        c = VOCAB.get(t, '')
        if c:
            chars.append(c)

    return ''.join(chars).strip()


def decode_npu_output(dat_path, output_scale=0.150270, output_zp=186):
    """Decode NPU uint8 output .dat file → text."""
    data = np.fromfile(dat_path, dtype=np.uint8)
    expected_size = SEQ_LEN * VOCAB_SIZE
    if len(data) != expected_size:
        print(f"  WARNING: output size {len(data)} != expected {expected_size}")
        return "[SIZE_MISMATCH]"

    # Dequantize
    uint8_data = data.reshape(SEQ_LEN, VOCAB_SIZE)
    logits = (uint8_data.astype(np.float32) - output_zp) * output_scale

    return ctc_greedy_decode(logits)


def parse_logcat_results(logcat_file):
    """Parse wav2vec2 results from logcat output."""
    results = {}
    pattern = re.compile(r'wav2vec2 result:.*transcription=\'([^\']*)\', confidence=')
    file_pattern = re.compile(r'Processing audio: (.+\.wav)')

    current_file = None
    with open(logcat_file, 'r') as f:
        for line in f:
            m = file_pattern.search(line)
            if m:
                current_file = m.group(1)
            m = pattern.search(line)
            if m and current_file:
                results[current_file] = m.group(1)
                current_file = None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', help='Directory with output_XXXX.dat files from vpm_run')
    parser.add_argument('--logcat-file', help='Logcat file with wav2vec2 results')
    parser.add_argument('--gt', required=True, help='Ground truth file (ground_truth.txt)')
    parser.add_argument('--output-scale', type=float, default=0.150270, help='NPU output dequantization scale')
    parser.add_argument('--output-zp', type=int, default=186, help='NPU output dequantization zero point')
    args = parser.parse_args()

    # Read ground truth
    gt = {}
    with open(args.gt, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                gt[parts[0]] = parts[1]

    print(f"Ground truth: {len(gt)} samples")

    # Get predictions
    predictions = {}

    if args.output_dir:
        # From vpm_run output .dat files
        for i, (wav_name, ref_text) in enumerate(gt.items()):
            dat_path = os.path.join(args.output_dir, f"output_{i:04d}.dat")
            if os.path.exists(dat_path):
                hyp = decode_npu_output(dat_path, args.output_scale, args.output_zp)
                predictions[wav_name] = hyp
            else:
                print(f"  Missing: {dat_path}")

    elif args.logcat_file:
        predictions = parse_logcat_results(args.logcat_file)

    else:
        print("ERROR: Specify --output-dir or --logcat-file")
        return

    # Evaluate
    total_cer_dist = 0
    total_cer_len = 0
    total_wer_dist = 0
    total_wer_len = 0
    exact_match = 0
    n_evaluated = 0

    print(f"\n{'#':>3} {'File':>25}  {'CER':>6}  {'WER':>6}  REF → HYP")
    print("-" * 120)

    for wav_name, ref_text in gt.items():
        if wav_name not in predictions:
            continue

        hyp_text = predictions[wav_name]
        ref_upper = ref_text.upper().strip()
        hyp_upper = hyp_text.upper().strip()

        # CER (character level, ignoring spaces)
        ref_nospace = ref_upper.replace(' ', '')
        hyp_nospace = hyp_upper.replace(' ', '')
        cer_dist = edit_distance(ref_nospace, hyp_nospace)
        cer = cer_dist / max(len(ref_nospace), 1) * 100

        # WER (word level)
        wer_dist, wer_len = word_edit_distance(ref_upper, hyp_upper)
        wer = wer_dist / max(wer_len, 1) * 100

        total_cer_dist += cer_dist
        total_cer_len += len(ref_nospace)
        total_wer_dist += wer_dist
        total_wer_len += wer_len

        if ref_upper == hyp_upper:
            exact_match += 1

        n_evaluated += 1

        # Truncate for display
        ref_disp = ref_upper[:40] + "..." if len(ref_upper) > 40 else ref_upper
        hyp_disp = hyp_upper[:40] + "..." if len(hyp_upper) > 40 else hyp_upper
        print(f"{n_evaluated:3d} {wav_name:>25}  {cer:5.1f}%  {wer:5.1f}%  {ref_disp} → {hyp_disp}")

    # Summary
    if n_evaluated > 0:
        avg_cer = total_cer_dist / max(total_cer_len, 1) * 100
        avg_wer = total_wer_dist / max(total_wer_len, 1) * 100
        print("-" * 120)
        print(f"\n=== Results ({n_evaluated} samples) ===")
        print(f"  CER: {avg_cer:.2f}% (edit_dist={total_cer_dist}, ref_chars={total_cer_len})")
        print(f"  WER: {avg_wer:.2f}% (edit_dist={total_wer_dist}, ref_words={total_wer_len})")
        print(f"  Exact match: {exact_match}/{n_evaluated} ({exact_match/n_evaluated*100:.1f}%)")
    else:
        print("\nNo samples evaluated!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute CER for Korean wav2vec2 NPU uint8 outputs vs ONNX float and GT text."""

import numpy as np
import os
import json
import re

# === Config ===
BASE = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_eager_op12/wksp/clean_pipeline"
OUTPUT_DIR = os.path.join(BASE, "ko_cer_outputs")
TEST_NPY_DIR = os.path.join(BASE, "test_npy")
MANIFEST = os.path.join(TEST_NPY_DIR, "manifest.txt")
ONNX_PATH = os.path.join(BASE, "wav2vec2_ko_eager_op12_3s.onnx")
VOCAB_PATH = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s/vocab.json"

# NPU output quantization params
OUT_SCALE = 0.06911206990480423
OUT_ZP = 74
PAD_ID = 53
NUM_SAMPLES = 50
SEQ_LEN = 149
VOCAB_SIZE = 56

# === Load vocab ===
with open(VOCAB_PATH) as f:
    vocab_dict = json.load(f)
id2char = {v: k for k, v in vocab_dict.items()}

# === Korean jamo composition ===
CHOSEONG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNGSEONG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONGSEONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

CHOSEONG_SET = set(CHOSEONG)
JUNGSEONG_SET = set(JUNGSEONG)
# Jongseong-only jamo (compound consonants that can't be choseong)
JONGSEONG_ONLY = set("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")

def compose_jamo(jamo_str):
    """Compose a sequence of Korean jamo into syllables."""
    result = []
    chars = list(jamo_str)
    i = 0
    while i < len(chars):
        c = chars[i]
        if c == ' ':
            result.append(' ')
            i += 1
            continue

        # Check if this is a choseong (initial consonant)
        if c in CHOSEONG_SET and i + 1 < len(chars) and chars[i+1] in JUNGSEONG_SET:
            cho = CHOSEONG.index(c)
            jung = JUNGSEONG.index(chars[i+1])
            jong = 0  # no final consonant by default

            # Check for jongseong
            if i + 2 < len(chars):
                next_c = chars[i+2]
                if next_c in JONGSEONG_ONLY:
                    # Compound consonant - always jongseong
                    jong = JONGSEONG.index(next_c)
                    i += 3
                elif next_c in CHOSEONG_SET:
                    # Could be jongseong of current or choseong of next
                    if i + 3 < len(chars) and chars[i+3] in JUNGSEONG_SET:
                        # Next char is choseong of next syllable
                        i += 2
                    else:
                        # It's jongseong of current syllable
                        if next_c in [JONGSEONG[j] for j in range(len(JONGSEONG)) if JONGSEONG[j]]:
                            jong = JONGSEONG.index(next_c)
                            i += 3
                        else:
                            i += 2
                else:
                    i += 2
            else:
                i += 2

            syllable = chr(0xAC00 + (cho * 21 + jung) * 28 + jong)
            result.append(syllable)
        else:
            # Can't compose, keep as-is
            result.append(c)
            i += 1

    return ''.join(result)

def ctc_greedy_decode(logits, blank_id=PAD_ID):
    """CTC greedy decoding: argmax, remove consecutive duplicates, remove blank."""
    token_ids = np.argmax(logits, axis=-1)
    # Remove consecutive duplicates
    prev = -1
    decoded = []
    for t in token_ids:
        if t != prev:
            if t != blank_id:
                decoded.append(int(t))
            prev = t
    return decoded

def tokens_to_text(token_ids):
    """Convert token IDs to Korean text."""
    jamo = ''.join(id2char.get(t, '') for t in token_ids)
    # Remove special tokens
    jamo = jamo.replace('<s>', '').replace('</s>', '').replace('[UNK]', '').replace('[PAD]', '')
    text = compose_jamo(jamo)
    return text

def edit_distance(s1, s2):
    """Compute Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def cer(hypothesis, reference):
    """Character Error Rate (ignoring spaces)."""
    hyp = hypothesis.replace(' ', '')
    ref = reference.replace(' ', '')
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return edit_distance(hyp, ref) / len(ref)

# === Load GT text ===
gt_texts = {}
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            idx = parts[0]  # e.g., "calib_0000"
            text = parts[1]
            num = int(idx.split('_')[1])
            gt_texts[num] = text

# === Decode NPU outputs ===
print("=" * 80)
print("Korean Wav2Vec2 uint8 NPU CER Measurement (50 samples)")
print("=" * 80)

npu_texts = {}
npu_cer_vs_gt = []
npu_nonpad_counts = []

for i in range(NUM_SAMPLES):
    dat_path = os.path.join(OUTPUT_DIR, f"test_output_{i:04d}.dat")
    if not os.path.exists(dat_path):
        print(f"Missing: {dat_path}")
        continue

    raw = np.fromfile(dat_path, dtype=np.uint8)
    assert raw.shape[0] == SEQ_LEN * VOCAB_SIZE, f"Unexpected size: {raw.shape[0]}"

    # Reshape and dequantize
    logits_u8 = raw.reshape(SEQ_LEN, VOCAB_SIZE)
    logits_f32 = (logits_u8.astype(np.float32) - OUT_ZP) * OUT_SCALE

    # CTC decode
    token_ids = ctc_greedy_decode(logits_f32)
    text = tokens_to_text(token_ids)
    npu_texts[i] = text

    # Count non-PAD tokens
    argmax_ids = np.argmax(logits_f32, axis=-1)
    nonpad = np.sum(argmax_ids != PAD_ID)
    npu_nonpad_counts.append(nonpad)

    # CER vs GT
    gt = gt_texts.get(i, "")
    c = cer(text, gt)
    npu_cer_vs_gt.append(c)

# === Run ONNX float inference for comparison ===
print("\nRunning ONNX float inference on same 50 inputs...")
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

    onnx_texts = {}
    onnx_cer_vs_gt = []

    for i in range(NUM_SAMPLES):
        npy_path = os.path.join(TEST_NPY_DIR, f"calib_{i:04d}.npy")
        if not os.path.exists(npy_path):
            continue

        audio = np.load(npy_path).astype(np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        outputs = sess.run(None, {"input_values": audio})
        logits = outputs[0]  # [1, 149, 56]
        logits = logits.squeeze(0)  # [149, 56]

        token_ids = ctc_greedy_decode(logits)
        text = tokens_to_text(token_ids)
        onnx_texts[i] = text

        gt = gt_texts.get(i, "")
        c = cer(text, gt)
        onnx_cer_vs_gt.append(c)

    # NPU vs ONNX CER
    npu_vs_onnx_cer = []
    for i in range(NUM_SAMPLES):
        if i in npu_texts and i in onnx_texts:
            c = cer(npu_texts[i], onnx_texts[i])
            npu_vs_onnx_cer.append(c)

    has_onnx = True
except Exception as e:
    print(f"ONNX inference failed: {e}")
    has_onnx = False

# === Print results ===
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

# Per-sample detail
print(f"\n{'#':>3} {'NonPAD':>6} {'NPU CER':>8}", end="")
if has_onnx:
    print(f" {'ONNX CER':>9} {'NPU/ONNX':>9}", end="")
print(f"  {'NPU Text':30s}  {'GT (first 30 chars)'}")
print("-" * 120)

for i in range(NUM_SAMPLES):
    npu_t = npu_texts.get(i, "")
    gt_t = gt_texts.get(i, "")
    nonpad = npu_nonpad_counts[i] if i < len(npu_nonpad_counts) else 0
    ncer = npu_cer_vs_gt[i] if i < len(npu_cer_vs_gt) else -1

    print(f"{i:3d} {nonpad:6d} {ncer:7.1%}", end="")
    if has_onnx:
        ocer = onnx_cer_vs_gt[i] if i < len(onnx_cer_vs_gt) else -1
        nvocer = npu_vs_onnx_cer[i] if i < len(npu_vs_onnx_cer) else -1
        print(f" {ocer:8.1%} {nvocer:8.1%}", end="")

    npu_short = npu_t[:30] if npu_t else "(empty)"
    gt_short = gt_t[:30] if gt_t else "(empty)"
    print(f"  {npu_short:30s}  {gt_short}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

avg_nonpad = np.mean(npu_nonpad_counts) if npu_nonpad_counts else 0
avg_npu_cer = np.mean(npu_cer_vs_gt) if npu_cer_vs_gt else 0
empty_count = sum(1 for t in npu_texts.values() if not t.strip())

print(f"  Samples:           {NUM_SAMPLES}")
print(f"  Avg non-PAD frames: {avg_nonpad:.1f} / {SEQ_LEN}")
print(f"  Empty outputs:     {empty_count}")
print(f"  NPU vs GT CER:     {avg_npu_cer:.2%}")

if has_onnx:
    avg_onnx_cer = np.mean(onnx_cer_vs_gt) if onnx_cer_vs_gt else 0
    avg_npu_vs_onnx = np.mean(npu_vs_onnx_cer) if npu_vs_onnx_cer else 0
    print(f"  ONNX vs GT CER:    {avg_onnx_cer:.2%}")
    print(f"  NPU vs ONNX CER:   {avg_npu_vs_onnx:.2%}  (quantization degradation)")

print(f"\nNote: GT text covers full utterance (6-15s), model only sees first 3s.")
print(f"      NPU vs ONNX CER is the best measure of quantization quality.")

# Also print: what fraction of GT is even coverable
if has_onnx:
    print(f"\n--- ONNX output samples (first 5) ---")
    for i in range(min(5, NUM_SAMPLES)):
        onnx_t = onnx_texts.get(i, "")
        gt_t = gt_texts.get(i, "")
        print(f"  [{i}] ONNX: {onnx_t}")
        print(f"       GT:   {gt_t}")
        print(f"       NPU:  {npu_texts.get(i, '')}")
        print()

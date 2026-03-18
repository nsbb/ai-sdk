#!/usr/bin/env python3
"""Fair CER: truncate GT to match the portion covered by 3s audio."""

import numpy as np
import os
import json

BASE = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_eager_op12/wksp/clean_pipeline"
OUTPUT_DIR = os.path.join(BASE, "ko_cer_outputs")
TEST_NPY_DIR = os.path.join(BASE, "test_npy")
MANIFEST = os.path.join(TEST_NPY_DIR, "manifest.txt")
ONNX_PATH = os.path.join(BASE, "wav2vec2_ko_eager_op12_3s.onnx")
VOCAB_PATH = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s/vocab.json"

OUT_SCALE = 0.06911206990480423
OUT_ZP = 74
PAD_ID = 53
SEQ_LEN = 149
VOCAB_SIZE = 56

with open(VOCAB_PATH) as f:
    vocab_dict = json.load(f)
id2char = {v: k for k, v in vocab_dict.items()}

CHOSEONG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNGSEONG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONGSEONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
CHOSEONG_SET = set(CHOSEONG)
JUNGSEONG_SET = set(JUNGSEONG)
JONGSEONG_ONLY = set("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")

def compose_jamo(jamo_str):
    result = []
    chars = list(jamo_str)
    i = 0
    while i < len(chars):
        c = chars[i]
        if c == ' ':
            result.append(' ')
            i += 1
            continue
        if c in CHOSEONG_SET and i + 1 < len(chars) and chars[i+1] in JUNGSEONG_SET:
            cho = CHOSEONG.index(c)
            jung = JUNGSEONG.index(chars[i+1])
            jong = 0
            if i + 2 < len(chars):
                next_c = chars[i+2]
                if next_c in JONGSEONG_ONLY:
                    jong = JONGSEONG.index(next_c)
                    i += 3
                elif next_c in CHOSEONG_SET:
                    if i + 3 < len(chars) and chars[i+3] in JUNGSEONG_SET:
                        i += 2
                    else:
                        if next_c in JONGSEONG:
                            jong = JONGSEONG.index(next_c)
                            i += 3
                        else:
                            i += 2
                else:
                    i += 2
            else:
                i += 2
            result.append(chr(0xAC00 + (cho * 21 + jung) * 28 + jong))
        else:
            result.append(c)
            i += 1
    return ''.join(result)

def ctc_greedy_decode(logits, blank_id=PAD_ID):
    token_ids = np.argmax(logits, axis=-1)
    prev = -1
    decoded = []
    for t in token_ids:
        if t != prev:
            if t != blank_id:
                decoded.append(int(t))
            prev = t
    return decoded

def tokens_to_text(token_ids):
    jamo = ''.join(id2char.get(t, '') for t in token_ids)
    jamo = jamo.replace('<s>', '').replace('</s>', '').replace('[UNK]', '').replace('[PAD]', '')
    return compose_jamo(jamo)

def edit_distance(s1, s2):
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

def cer(hyp, ref):
    h = hyp.replace(' ', '')
    r = ref.replace(' ', '')
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(h, r) / len(r)

# Load GT
gt_texts = {}
gt_durations = {}
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) >= 3:
            num = int(parts[0].split('_')[1])
            gt_texts[num] = parts[1]
            dur = float(parts[2].replace('s', ''))
            gt_durations[num] = dur

# Truncate GT to 3-second portion
def truncate_gt(gt_text, full_dur, target_dur=3.0):
    """Estimate how many characters correspond to the first target_dur seconds."""
    if full_dur <= target_dur:
        return gt_text
    ratio = target_dur / full_dur
    # Remove spaces for character counting, then map back
    chars_no_space = gt_text.replace(' ', '')
    n_chars = max(1, int(len(chars_no_space) * ratio))
    # Reconstruct with spaces
    count = 0
    cut_pos = len(gt_text)
    for idx, c in enumerate(gt_text):
        if c != ' ':
            count += 1
        if count >= n_chars:
            cut_pos = idx + 1
            break
    return gt_text[:cut_pos]

# ONNX inference
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

print("=" * 100)
print("Fair CER Measurement — GT truncated to 3-second estimate")
print("=" * 100)
print(f"\n{'#':>3} {'Dur':>5} {'ONNX/GT':>8} {'NPU/GT':>7} {'NPU/ONNX':>9}  {'ONNX':30s} {'GT(3s)':30s} {'NPU'}")
print("-" * 140)

onnx_cers = []
npu_cers = []
npu_vs_onnx_cers = []

for i in range(50):
    # ONNX inference
    npy_path = os.path.join(TEST_NPY_DIR, f"calib_{i:04d}.npy")
    audio = np.load(npy_path).astype(np.float32).reshape(1, -1)
    logits_onnx = sess.run(None, {"input_values": audio})[0].squeeze(0)
    onnx_tokens = ctc_greedy_decode(logits_onnx)
    onnx_text = tokens_to_text(onnx_tokens)

    # NPU output
    dat_path = os.path.join(OUTPUT_DIR, f"test_output_{i:04d}.dat")
    raw = np.fromfile(dat_path, dtype=np.uint8).reshape(SEQ_LEN, VOCAB_SIZE)
    logits_npu = (raw.astype(np.float32) - OUT_ZP) * OUT_SCALE
    npu_tokens = ctc_greedy_decode(logits_npu)
    npu_text = tokens_to_text(npu_tokens)

    # Truncated GT
    gt_full = gt_texts.get(i, "")
    dur = gt_durations.get(i, 3.0)
    gt_3s = truncate_gt(gt_full, dur)

    # CERs
    onnx_c = cer(onnx_text, gt_3s)
    npu_c = cer(npu_text, gt_3s)
    nvo_c = cer(npu_text, onnx_text)

    onnx_cers.append(onnx_c)
    npu_cers.append(npu_c)
    npu_vs_onnx_cers.append(nvo_c)

    print(f"{i:3d} {dur:5.1f} {onnx_c:7.1%} {npu_c:6.1%} {nvo_c:8.1%}  {onnx_text[:30]:30s} {gt_3s[:30]:30s} {npu_text[:30]}")

# Summary
print("\n" + "=" * 100)
print("SUMMARY (GT truncated to ~3s estimate)")
print("=" * 100)
print(f"  ONNX float vs GT(3s):  {np.mean(onnx_cers):.2%}")
print(f"  NPU uint8 vs GT(3s):   {np.mean(npu_cers):.2%}")
print(f"  NPU uint8 vs ONNX:     {np.mean(npu_vs_onnx_cers):.2%}  ← quantization degradation")
print()
print(f"  비교: KoCitrinet 300f int8 CER = 44.44% (3초 입력, 120ms)")
print(f"        Wav2Vec2-EN uint8 CER ≈ 25% (5초 입력, 720ms)")

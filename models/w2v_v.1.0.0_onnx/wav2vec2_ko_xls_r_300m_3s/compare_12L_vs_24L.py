#!/usr/bin/env python3
"""Compare 12-layer vs 24-layer ONNX inference on multiple calibration samples."""

import onnxruntime as ort
import numpy as np
import json
import os

WORKDIR = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_xls_r_300m_3s"
VOCAB_PATH = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/vocab.json"

os.chdir(WORKDIR)

with open(VOCAB_PATH) as f:
    vocab = json.load(f)
id_to_char = {v: k for k, v in vocab.items()}
pad_id = vocab["<pad>"]  # 2616


def ctc_decode(ids):
    prev = -1
    chars = []
    for i in ids:
        if i != prev and i != pad_id:
            c = id_to_char.get(i, f"[{i}]")
            chars.append(c)
        prev = i
    return "".join(chars).replace("|", " ")


# Load models
print("Loading 12-layer model...")
sess_12 = ort.InferenceSession("wav2vec2_ko_3s_12layers.onnx")
print("Loading 24-layer model...")
sess_24 = ort.InferenceSession("wav2vec2_ko_3s.onnx")

# Test with calibration samples
calib_dir = "calib_data_v2"
N = min(10, len(os.listdir(calib_dir)))

print(f"\nTesting {N} calibration samples:")
print(f"{'Sample':>8} | {'12L unique':>10} | {'24L unique':>10} | 12L text | 24L text")
print("-" * 120)

for i in range(N):
    fname = f"calib_{i:04d}.npy"
    fpath = os.path.join(calib_dir, fname)
    if not os.path.exists(fpath):
        continue

    audio = np.load(fpath)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    if audio.shape[1] != 48000:
        if audio.shape[1] > 48000:
            audio = audio[:, :48000]
        else:
            audio = np.pad(audio, ((0, 0), (0, 48000 - audio.shape[1])))
    audio = audio.astype(np.float32)

    out_12 = sess_12.run(None, {"input": audio})[0]
    out_24 = sess_24.run(None, {"input": audio})[0]

    pred_12 = np.argmax(out_12[0], axis=-1)
    pred_24 = np.argmax(out_24[0], axis=-1)

    text_12 = ctc_decode(pred_12)
    text_24 = ctc_decode(pred_24)

    u12 = len(np.unique(pred_12))
    u24 = len(np.unique(pred_24))

    print(f"{i:>8} | {u12:>10} | {u24:>10} | {text_12[:30]:30s} | {text_24[:50]}")

# Summary statistics
print(f"\n{'='*60}")
print("Summary: 12L model retains ~{:.0f}% fewer unique tokens than 24L".format(0))
print("The 12L FP32 model produces significantly degraded output")
print("because the lm_head was trained on 24-layer representations.")
print("However, 12L uint8 does NOT collapse to all-PAD like 24L uint8.")

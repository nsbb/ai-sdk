#!/usr/bin/env python3
"""Compare 12-layer pruned model vs 24-layer original on real audio."""

import onnxruntime as ort
import numpy as np
import json
import os

WORKDIR = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_xls_r_300m_3s"
VOCAB_PATH = os.path.join(WORKDIR, "..", "wav2vec2_ko_base_3s", "vocab.json")

os.chdir(WORKDIR)

# Load vocab
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
id_to_char = {v: k for k, v in vocab.items()}
pad_id = vocab.get("<pad>", 0)
print(f"Vocab size: {len(vocab)}, PAD ID: {pad_id}")


def ctc_decode(ids, blank_id=0):
    prev = -1
    chars = []
    for i in ids:
        if i != prev and i != blank_id:
            c = id_to_char.get(i, f"[{i}]")
            chars.append(c)
        prev = i
    return "".join(chars).replace("|", " ")


# Load test audio
audio = np.load("test_audio.npy")
print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
if audio.ndim == 1:
    audio = audio.reshape(1, -1)
if audio.shape[1] != 48000:
    print(f"Adjusting from {audio.shape[1]} to 48000")
    if audio.shape[1] > 48000:
        audio = audio[:, :48000]
    else:
        audio = np.pad(audio, ((0, 0), (0, 48000 - audio.shape[1])))
audio = audio.astype(np.float32)

# Load models
print("\nLoading 12-layer model...")
sess_12 = ort.InferenceSession("wav2vec2_ko_3s_12layers.onnx")
print("Loading 24-layer model...")
sess_24 = ort.InferenceSession("wav2vec2_ko_3s.onnx")

# Run inference
import time

t0 = time.time()
out_12 = sess_12.run(None, {"input": audio})[0]
t12 = time.time() - t0

t0 = time.time()
out_24 = sess_24.run(None, {"input": audio})[0]
t24 = time.time() - t0

print(f"\n12-layer: {t12*1000:.0f}ms, 24-layer: {t24*1000:.0f}ms")

# Decode
pred_12 = np.argmax(out_12[0], axis=-1)
pred_24 = np.argmax(out_24[0], axis=-1)

text_12 = ctc_decode(pred_12, blank_id=pad_id)
text_24 = ctc_decode(pred_24, blank_id=pad_id)

print(f"\n12-layer output: \"{text_12}\"")
print(f"24-layer output: \"{text_24}\"")
print(f"\n12-layer unique tokens: {len(np.unique(pred_12))}")
print(f"24-layer unique tokens: {len(np.unique(pred_24))}")
print(f"12-layer pred IDs: {pred_12[:40]}")
print(f"24-layer pred IDs: {pred_24[:40]}")

# Also test with a few calibration samples
print("\n=== Calibration sample tests ===")
calib_dir = "calib_data_v2"
for i in range(3):
    fname = f"calib_{i:04d}.npy"
    fpath = os.path.join(calib_dir, fname)
    if os.path.exists(fpath):
        audio_c = np.load(fpath)
        if audio_c.ndim == 1:
            audio_c = audio_c.reshape(1, -1)
        if audio_c.shape[1] != 48000:
            if audio_c.shape[1] > 48000:
                audio_c = audio_c[:, :48000]
            else:
                audio_c = np.pad(audio_c, ((0, 0), (0, 48000 - audio_c.shape[1])))
        audio_c = audio_c.astype(np.float32)

        o12 = sess_12.run(None, {"input": audio_c})[0]
        o24 = sess_24.run(None, {"input": audio_c})[0]
        p12 = np.argmax(o12[0], axis=-1)
        p24 = np.argmax(o24[0], axis=-1)
        t12 = ctc_decode(p12, blank_id=pad_id)
        t24 = ctc_decode(p24, blank_id=pad_id)
        print(f"\nSample {i}:")
        print(f"  12L: \"{t12}\" (unique={len(np.unique(p12))})")
        print(f"  24L: \"{t24}\" (unique={len(np.unique(p24))})")

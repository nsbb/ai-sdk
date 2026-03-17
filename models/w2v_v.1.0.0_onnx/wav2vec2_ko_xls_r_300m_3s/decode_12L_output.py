#!/usr/bin/env python3
"""Decode Acuity inference outputs from 12-layer pruned model."""

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


def load_acuity_tensor(path):
    """Load Acuity .tensor file (text format)."""
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals = [float(x) for x in line.split()]
                values.extend(vals)
            except ValueError:
                continue
    return np.array(values, dtype=np.float32)


def ctc_decode(ids, blank_id=0):
    prev = -1
    chars = []
    for i in ids:
        if i != prev and i != blank_id:
            c = id_to_char.get(i, f"[{i}]")
            chars.append(c)
        prev = i
    return "".join(chars).replace("|", " ")


EXPECTED = 149 * 2617

# Process each output directory
for label, dirpath in [
    ("12L FP32", "inf_12L_float"),
    ("12L uint8", "inf_12L_uint8"),
    ("24L uint8", "inf_uint8_test"),
    ("24L pcq", "inf_pcq_test"),
    ("24L FP32", "inf_float_test"),
]:
    print(f"\n=== {label} ===")
    if not os.path.exists(dirpath):
        print("  (directory not found)")
        continue

    tensor_file = None
    for f in sorted(os.listdir(dirpath)):
        if f.endswith(".tensor") and "out0" in f and "input" not in f:
            tensor_file = os.path.join(dirpath, f)
            break

    if not tensor_file:
        print("  (no output tensor found)")
        continue

    data = load_acuity_tensor(tensor_file)
    print(f"  Data size: {len(data)} (expected {EXPECTED})")

    if len(data) == EXPECTED:
        logits = data.reshape(149, 2617)
        pred = np.argmax(logits, axis=-1)
        text = ctc_decode(pred, blank_id=pad_id)

        unique_count = len(np.unique(pred))
        pad_count = np.sum(pred == pad_id)
        t2616_count = np.sum(pred == 2616)

        print(f"  Unique tokens: {unique_count}")
        print(f"  PAD (id={pad_id}) count: {pad_count}/{len(pred)}")
        print(f"  Token 2616 count: {t2616_count}/{len(pred)}")
        print(f"  Predicted IDs (first 40): {pred[:40]}")
        print(f"  Decoded: \"{text}\"")

        # Logit statistics
        print(f"  Logit range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"  Logit mean: {logits.mean():.4f}")
    else:
        print(f"  Size mismatch!")

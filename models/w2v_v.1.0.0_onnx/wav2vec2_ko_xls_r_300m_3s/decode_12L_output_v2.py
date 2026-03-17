#!/usr/bin/env python3
"""Decode Acuity inference outputs using full 2617-token Korean vocab."""

import numpy as np
import json
import os

WORKDIR = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_xls_r_300m_3s"
VOCAB_PATH = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/vocab.json"

os.chdir(WORKDIR)

# Load full vocab (2617 tokens)
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
id_to_char = {v: k for k, v in vocab.items()}
pad_id = vocab.get("<pad>", 2616)
blank_id = vocab.get("|", 4)  # word boundary
print(f"Full vocab size: {len(vocab)}, PAD ID: {pad_id}, blank/WB ID: {blank_id}")


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


def ctc_decode(ids, pad_id=2616):
    """CTC greedy decode: collapse repeats and remove PAD."""
    prev = -1
    chars = []
    for i in ids:
        if i != prev and i != pad_id:
            c = id_to_char.get(i, f"[{i}]")
            chars.append(c)
        prev = i
    return "".join(chars).replace("|", " ")


EXPECTED = 149 * 2617

# Process each output directory
for label, dirpath in [
    ("12L FP32", "inf_12L_float"),
    ("12L uint8", "inf_12L_uint8"),
    ("24L FP32", "inf_float_test"),
    ("24L uint8", "inf_uint8_test"),
    ("24L pcq", "inf_pcq_test"),
]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
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

    if len(data) == EXPECTED:
        logits = data.reshape(149, 2617)
        pred = np.argmax(logits, axis=-1)
        text = ctc_decode(pred, pad_id=pad_id)

        unique_count = len(np.unique(pred))
        pad_count = np.sum(pred == pad_id)
        non_pad = pred[pred != pad_id]

        print(f"  Unique tokens: {unique_count}")
        print(f"  PAD frames: {pad_count}/{len(pred)} ({pad_count/len(pred)*100:.1f}%)")
        print(f"  Non-PAD tokens: {list(non_pad)}")
        if len(non_pad) > 0:
            non_pad_chars = [id_to_char.get(int(t), f"[{t}]") for t in non_pad]
            print(f"  Non-PAD chars: {non_pad_chars}")
        print(f"  Decoded text: \"{text}\"")
        print(f"  Logit range: [{logits.min():.4f}, {logits.max():.4f}]")
    else:
        print(f"  Size mismatch: {len(data)} (expected {EXPECTED})")

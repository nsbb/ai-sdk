#!/usr/bin/env python3
"""Decode multiple Acuity inference outputs (iter_0..iter_N) for comparison."""

import numpy as np
import json
import os
import glob

WORKDIR = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_xls_r_300m_3s"
VOCAB_PATH = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/vocab.json"

os.chdir(WORKDIR)

with open(VOCAB_PATH) as f:
    vocab = json.load(f)
id_to_char = {v: k for k, v in vocab.items()}
pad_id = vocab["<pad>"]  # 2616


def load_acuity_tensor(path):
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


def ctc_decode(ids):
    prev = -1
    chars = []
    for i in ids:
        if i != prev and i != pad_id:
            c = id_to_char.get(i, f"[{i}]")
            chars.append(c)
        prev = i
    return "".join(chars).replace("|", " ")


EXPECTED = 149 * 2617

for label, dirpath in [
    ("12L FP32 (200-iter calib)", "inf_12L_float_200"),
    ("12L uint8 (200-iter calib)", "inf_12L_uint8_200"),
]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if not os.path.exists(dirpath):
        print("  (not found)")
        continue

    # Find all output tensors
    files = sorted(glob.glob(os.path.join(dirpath, "iter_*_out0_*.tensor")))
    if not files:
        # Try alternate naming
        files = sorted(glob.glob(os.path.join(dirpath, "iter_*_attach_*.tensor")))

    for tensor_file in files:
        fname = os.path.basename(tensor_file)
        # Extract iteration number
        iter_num = fname.split("_")[1]

        data = load_acuity_tensor(tensor_file)
        if len(data) != EXPECTED:
            continue

        logits = data.reshape(149, 2617)
        pred = np.argmax(logits, axis=-1)
        text = ctc_decode(pred)

        pad_count = np.sum(pred == pad_id)
        unique = len(np.unique(pred))
        non_pad = pred[pred != pad_id]

        print(f"  iter {iter_num}: unique={unique:2d}, "
              f"PAD={pad_count:3d}/149, "
              f"text=\"{text}\"")

#!/usr/bin/env python3
"""Decode Acuity .tensor file (text format)."""
import numpy as np
import json
import sys

with open("/home/nsbb/travail/wav2vec/w2v_v.1.0.0_onnx/vocab.json") as f:
    vocab_raw = json.load(f)
vocab = {v: k for k, v in vocab_raw.items()}
pad_id = 2616

tensor_path = sys.argv[1]
vals = np.loadtxt(tensor_path).reshape(149, 2617)
tokens = np.argmax(vals, axis=1)

deduped = [tokens[0]]
for t in tokens[1:]:
    if t != deduped[-1]:
        deduped.append(t)

text = ''.join(vocab.get(int(t), '') for t in deduped if int(t) not in {0, pad_id})
text = text.replace('|', ' ')

non_pad = sum(1 for t in tokens if t != pad_id)
print(f'Text: "{text}"')
print(f'Non-pad: {non_pad}/149')
print(f'Logit range: [{vals.min():.2f}, {vals.max():.2f}]')
print(f'Pad logit mean: {vals[:, pad_id].mean():.2f}')

#!/usr/bin/env python3
"""Decode Korean Wav2Vec2 NPU uint8 output."""
import numpy as np
import json
import sys

# Load vocab
with open("/home/nsbb/travail/wav2vec/w2v_v.1.0.0_onnx/vocab.json") as f:
    vocab_raw = json.load(f)
vocab = {v: k for k, v in vocab_raw.items()}
pad_id = vocab_raw.get("<pad>", 2616)

# NPU output params
SCALE = 0.219490
ZP = 122
SEQ_LEN = 149
VOCAB_SIZE = 2617

dat_path = sys.argv[1] if len(sys.argv) > 1 else "output_0.dat"
data = np.fromfile(dat_path, dtype=np.uint8)
print(f"Output size: {len(data)} bytes (expected {SEQ_LEN * VOCAB_SIZE})")

def ctc_decode(logits_2d):
    tokens = np.argmax(logits_2d, axis=1)
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)
    blank_ids = {0, pad_id}
    text = ''.join(vocab.get(int(t), '') for t in deduped if int(t) not in blank_ids)
    return text.replace('|', ' '), tokens

# Try [seq, vocab] layout
logits = data[:SEQ_LEN * VOCAB_SIZE].reshape(SEQ_LEN, VOCAB_SIZE)
dequant = (logits.astype(np.float32) - ZP) * SCALE
text1, tokens1 = ctc_decode(dequant)
print(f"\n[seq,vocab] layout: '{text1}'")
print(f"Top tokens: {tokens1[:30]}")
print(f"Unique tokens: {len(set(tokens1))}")

# Try [vocab, seq] layout (transposed)
logits_t = data[:SEQ_LEN * VOCAB_SIZE].reshape(VOCAB_SIZE, SEQ_LEN)
dequant_t = (logits_t.astype(np.float32) - ZP) * SCALE
text2, tokens2 = ctc_decode(dequant_t.T)
print(f"\n[vocab,seq] layout: '{text2}'")
print(f"Top tokens: {tokens2[:30]}")
print(f"Unique tokens: {len(set(tokens2))}")

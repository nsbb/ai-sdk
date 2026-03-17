#!/usr/bin/env python3
"""Analyze NPU output to understand what's happening."""
import numpy as np

SCALE = 0.219490
ZP = 122
SEQ_LEN = 149
VOCAB_SIZE = 2617

data = np.fromfile("output_0.dat", dtype=np.uint8)
logits = data.reshape(SEQ_LEN, VOCAB_SIZE)

print("=== Raw uint8 analysis ===")
print(f"Min: {logits.min()}, Max: {logits.max()}, Mean: {logits.mean():.1f}")
print(f"Unique values: {len(np.unique(logits))}")
print(f"Value histogram (top 10):")
vals, counts = np.unique(logits, return_counts=True)
idx = np.argsort(-counts)
for i in idx[:10]:
    print(f"  uint8={vals[i]:3d}, count={counts[i]:6d}, dequant={((vals[i]-ZP)*SCALE):+.3f}")

print(f"\n=== Per-position analysis ===")
dequant = (logits.astype(np.float32) - ZP) * SCALE
for pos in [0, 10, 20, 50, 100, 148]:
    row = dequant[pos]
    top5_idx = np.argsort(-row)[:5]
    print(f"Pos {pos:3d}: argmax={np.argmax(row):4d}, max={row.max():+.3f}, "
          f"top5_idx={top5_idx.tolist()}, top5_val=[{', '.join(f'{row[i]:+.3f}' for i in top5_idx)}]")

print(f"\n=== Is output constant? ===")
# Check if all rows are identical
all_same = all(np.array_equal(logits[0], logits[i]) for i in range(1, SEQ_LEN))
print(f"All rows identical: {all_same}")

# Check specific indices
print(f"\nRow 0 first 20 values: {logits[0, :20].tolist()}")
print(f"Row 0 last 20 values: {logits[0, -20:].tolist()}")
print(f"Row 50 first 20 values: {logits[50, :20].tolist()}")

# Check the pad position (2616)
print(f"\nPad position (2616) values across seq:")
print(f"  {logits[:20, 2616].tolist()}")

# Check if it's a calibration issue - compare with Acuity FP32 simulation
print(f"\n=== Argmax distribution ===")
argmax_per_pos = np.argmax(dequant, axis=1)
unique_tokens, token_counts = np.unique(argmax_per_pos, return_counts=True)
for t, c in zip(unique_tokens, token_counts):
    print(f"  Token {t}: {c} times")

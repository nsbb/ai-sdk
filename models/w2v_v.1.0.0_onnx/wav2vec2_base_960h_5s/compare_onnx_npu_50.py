#!/usr/bin/env python3
"""Compare ONNX FP32 vs NPU uint8 for 50 samples."""
import numpy as np
import onnxruntime as ort

ONNX_PATH = "wav2vec2_base_960h_5s.onnx"
VOCAB = {0:'', 1:'', 2:'', 3:'', 4:' ', 5:'E', 6:'T', 7:'A', 8:'O', 9:'N',
         10:'I', 11:'H', 12:'S', 13:'R', 14:'D', 15:'L', 16:'U', 17:'M', 18:'W', 19:'C',
         20:'F', 21:'G', 22:'Y', 23:'P', 24:'B', 25:'V', 26:'K', 27:"'", 28:'X', 29:'J',
         30:'Q', 31:'Z'}
SEQ_LEN = 249
VOCAB_SIZE = 32

def ctc_decode(logits_2d):
    tokens = np.argmax(logits_2d, axis=1)
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)
    return ''.join(VOCAB.get(int(t), '') for t in deduped if VOCAB.get(int(t), '')).strip()

def npu_decode(dat_path, scale=0.150270, zp=186):
    data = np.fromfile(dat_path, dtype=np.uint8).reshape(SEQ_LEN, VOCAB_SIZE)
    logits = (data.astype(np.float32) - zp) * scale
    return ctc_decode(logits)

def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

# Read ground truth
gt = {}
with open("data/english_test/ground_truth.txt") as f:
    for line in f:
        if line.startswith("#"): continue
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            gt[parts[0]] = parts[1]

sess = ort.InferenceSession(ONNX_PATH)

total_onnx_cer_dist = 0
total_onnx_cer_len = 0
total_npu_cer_dist = 0
total_npu_cer_len = 0
total_quant_cer_dist = 0
total_quant_cer_len = 0
onnx_exact = 0
npu_exact = 0

for i, (wav_name, ref_text) in enumerate(gt.items()):
    npy_path = f"data/english_test/en_test_{i:04d}.npy"
    dat_path = f"data/english_test/outputs/output_{i:04d}.dat"

    audio = np.load(npy_path).astype(np.float32)
    logits = sess.run(None, {"input_values": audio})[0]
    onnx_text = ctc_decode(logits[0])
    npu_text = npu_decode(dat_path)

    ref_ns = ref_text.upper().replace(' ', '')
    onnx_ns = onnx_text.replace(' ', '')
    npu_ns = npu_text.replace(' ', '')

    # ONNX vs GT
    onnx_cer_d = edit_distance(ref_ns, onnx_ns)
    total_onnx_cer_dist += onnx_cer_d
    total_onnx_cer_len += len(ref_ns)
    if ref_text.upper() == onnx_text: onnx_exact += 1

    # NPU vs GT
    npu_cer_d = edit_distance(ref_ns, npu_ns)
    total_npu_cer_dist += npu_cer_d
    total_npu_cer_len += len(ref_ns)
    if ref_text.upper() == npu_text: npu_exact += 1

    # NPU vs ONNX (quantization degradation)
    quant_d = edit_distance(onnx_ns, npu_ns)
    total_quant_cer_dist += quant_d
    total_quant_cer_len += len(onnx_ns)

onnx_cer = total_onnx_cer_dist / max(total_onnx_cer_len, 1) * 100
npu_cer = total_npu_cer_dist / max(total_npu_cer_len, 1) * 100
quant_cer = total_quant_cer_dist / max(total_quant_cer_len, 1) * 100

print(f"=== 50-Sample CER Evaluation ===")
print(f"")
print(f"ONNX FP32 vs Ground Truth:  CER {onnx_cer:.2f}%  (exact {onnx_exact}/50)")
print(f"NPU uint8 vs Ground Truth:  CER {npu_cer:.2f}%  (exact {npu_exact}/50)")
print(f"NPU uint8 vs ONNX FP32:     CER {quant_cer:.2f}%  (quantization degradation)")
print(f"")
print(f"Degradation: +{npu_cer - onnx_cer:.2f}%p")

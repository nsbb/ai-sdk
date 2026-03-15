#!/usr/bin/env python3
"""Compare ONNX FP32 vs NPU uint8 output for CER evaluation."""
import numpy as np
import onnxruntime as ort
import soundfile as sf

ONNX_PATH = "wav2vec2_base_960h_5s.onnx"
VOCAB = {0:'', 1:'', 2:'', 3:'', 4:' ', 5:'E', 6:'T', 7:'A', 8:'O', 9:'N',
         10:'I', 11:'H', 12:'S', 13:'R', 14:'D', 15:'L', 16:'U', 17:'M', 18:'W', 19:'C',
         20:'F', 21:'G', 22:'Y', 23:'P', 24:'B', 25:'V', 26:'K', 27:"'", 28:'X', 29:'J',
         30:'Q', 31:'Z'}

def ctc_decode(logits_2d):
    tokens = np.argmax(logits_2d, axis=1)
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)
    return ''.join(VOCAB.get(int(t), '') for t in deduped if VOCAB.get(int(t), '')).strip()

def npu_decode(dat_path, scale=0.150270, zp=186):
    data = np.fromfile(dat_path, dtype=np.uint8).reshape(249, 32)
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

sess = ort.InferenceSession(ONNX_PATH)

files = [
    ("data/test.wav", "1188-133604-0000"),
    ("/home/nsbb/travail/claude/T527/ai-sdk/models/deepspeech2/data/1188-133604-0010.flac.wav", "1188-133604-0010"),
    ("/home/nsbb/travail/claude/T527/ai-sdk/models/deepspeech2/data/1188-133604-0025.flac.wav", "1188-133604-0025"),
]

known_gt_0 = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"

print("=== ONNX FP32 vs NPU uint8 Comparison ===\n")

total_cer_dist = 0
total_cer_len = 0

for i, (path, uid) in enumerate(files):
    audio, sr = sf.read(path, dtype='float32')
    if len(audio) > 80000:
        audio = audio[:80000]
    else:
        audio = np.pad(audio, (0, 80000 - len(audio)))

    # ONNX
    inp = audio.reshape(1, -1).astype(np.float32)
    logits = sess.run(None, {"input_values": inp})[0]
    onnx_text = ctc_decode(logits[0])

    # NPU
    npu_text = npu_decode(f"data/english_test/outputs/output_{i:04d}.dat")

    # CER: NPU vs ONNX
    ref_ns = onnx_text.replace(' ', '')
    hyp_ns = npu_text.replace(' ', '')
    cer_dist = edit_distance(ref_ns, hyp_ns)
    cer = cer_dist / max(len(ref_ns), 1) * 100

    total_cer_dist += cer_dist
    total_cer_len += len(ref_ns)

    print(f"[{i}] {uid}")
    print(f"    ONNX: {onnx_text}")
    print(f"    NPU:  {npu_text}")
    print(f"    CER(NPU vs ONNX): {cer:.1f}%")

    if i == 0:
        gt_ns = known_gt_0.replace(' ', '')
        onnx_cer = edit_distance(gt_ns, ref_ns) / len(gt_ns) * 100
        npu_cer = edit_distance(gt_ns, hyp_ns) / len(gt_ns) * 100
        print(f"    GT:   {known_gt_0}")
        print(f"    ONNX vs GT: CER {onnx_cer:.1f}%")
        print(f"    NPU  vs GT: CER {npu_cer:.1f}%")
    print()

avg_cer = total_cer_dist / max(total_cer_len, 1) * 100
print(f"=== Overall NPU vs ONNX CER: {avg_cer:.2f}% ===")
print(f"    (This measures quantization degradation)")

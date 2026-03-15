#!/bin/bash
# Priority-ordered test of Korean wav2vec2 NB variants on T527 NPU
# Tests the most promising variants first (Phase 4 opset12 re-exports)

set -e
WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
WORK_DIR="/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s"
DEVICE_DIR="/data/local/tmp/w2v_ko_test"
AUDIO_NPY="${WORK_DIR}/ko_calib_npy/ko_calib_0000.npy"
RESULTS="${WORK_DIR}/npu_test_results.txt"

echo "=== Korean wav2vec2 NPU Priority Test ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "" | tee "$RESULTS"

# Check device
$WIN_ADB devices | grep -q "device$" || { echo "ERROR: No device found"; exit 1; }
$WIN_ADB shell "mkdir -p $DEVICE_DIR" 2>/dev/null

# Priority order (most likely to work first)
declare -a PRIORITY_NBS=(
    "nopad10_opset12_sim_ma"       # 1: opset12+sim, 1 var, uint8 MA — BEST CANDIDATE
    "nopad10_opset12_ma"           # 2: opset12, 12 var (same as English), uint8 MA
    "nopad10_opset12_sim_int16"    # 3: opset12+sim, int16 DFP — highest accuracy
    "nopad10_sim_ma"               # 4: onnxsim on opset14, uint8 MA
    "nopad10_opset12_sim_int8"     # 5: opset12+sim, int8 signed
    "nopad10_opset12_sim_hybrid_ma" # 6: hybrid uint8+int16
    "clip3s_nopad10_opset12_sim_ma" # 7: weight-clipped uint8 MA (98MB)
    "clip3s_nopad10_opset12_sim_int16" # 8: weight-clipped int16 DFP (175MB)
    "nopad10_opset12_sim_kl"       # 9: KL divergence
    "nopad10_opset12_sim_normal"   # 10: normal quantization
    "nopad10_opset12_sim_ma001"    # 11: MA weight 0.001
    "nopad10_opset12_sim_ma01"     # 12: MA weight 0.01
    "nopad10_opset12_sim_ma96"     # 13: 96 calibration samples
    "nopad10_ma"                   # 14: original opset14 (known garbled)
    "6L_nopad10_opset12_sim_ma"    # 15: 6-layer (sim=garbage, skip if #1 works)
    "relu_nopad10_opset12_sim_ma"  # 16: ReLU (sim=garbage)
    "2L_nopad10_opset12_sim_ma"    # 17: 2-layer canary (15MB)
    "combo_sim_relu_6l_nopad10_ma" # 18: combo
    "6L_relu_nopad10_opset12_sim_ma" # 19: smallest
)

for name in "${PRIORITY_NBS[@]}"; do
    nb_path="${WORK_DIR}/wksp/wav2vec2_ko_base_3s_${name}_nbg_unify/network_binary.nb"
    meta_path="${WORK_DIR}/wksp/wav2vec2_ko_base_3s_${name}_nbg_unify/nbg_meta.json"

    if [ ! -f "$nb_path" ]; then
        echo "SKIP: $name (NB not found)" | tee -a "$RESULTS"
        continue
    fi

    nb_size=$(ls -lh "$nb_path" | awk '{print $5}')
    echo "--- [$name] ${nb_size} ---" | tee -a "$RESULTS"

    # Generate quantized input based on metadata
    python3 -c "
import numpy as np, json, sys
audio = np.load('$AUDIO_NPY').astype(np.float32).flatten()[:48000]
if len(audio) < 48000: audio = np.pad(audio, (0, 48000-len(audio)))
meta = json.load(open('$meta_path'))
inp = list(meta['Inputs'].values())[0]['quantize']
qtype = inp.get('qtype', 'u8')
if qtype == 'i16':
    fl = inp['fl']
    q = np.clip(np.round(audio * (2**fl)), -32768, 32767).astype(np.int16)
    q.tofile('/tmp/input_0_test.dat')
    print(f'  Input: i16 fl={fl}', file=sys.stderr)
elif qtype == 'i8':
    scale, zp = inp['scale'], inp['zero_point']
    q = np.clip(np.round(audio / scale + zp), -128, 127).astype(np.int8)
    q.tofile('/tmp/input_0_test.dat')
    print(f'  Input: i8 scale={scale:.6f} zp={zp}', file=sys.stderr)
else:
    scale, zp = inp['scale'], inp['zero_point']
    q = np.clip(np.round(audio / scale + zp), 0, 255).astype(np.uint8)
    q.tofile('/tmp/input_0_test.dat')
    print(f'  Input: u8 scale={scale:.6f} zp={zp}', file=sys.stderr)
" 2>&1

    # Push NB and input
    $WIN_ADB push "$nb_path" "$DEVICE_DIR/network_binary.nb" > /dev/null 2>&1
    $WIN_ADB push /tmp/input_0_test.dat "$DEVICE_DIR/input_0.dat" > /dev/null 2>&1
    $WIN_ADB shell "echo 'input_0.dat' > $DEVICE_DIR/sample.txt" 2>/dev/null
    sleep 1

    # Run vpm_run
    result=$(timeout 30 $WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)

    if echo "$result" | grep -q "fail to run\|status=-1"; then
        echo "  >>> FAILED (NPU error)" | tee -a "$RESULTS"
        echo "" | tee -a "$RESULTS"
        continue
    fi

    inference_time=$(echo "$result" | grep "run time" | grep -oP '\d+(?= us)')
    echo "  Inference: ${inference_time}us" | tee -a "$RESULTS"

    # Pull and decode output
    output_file="${WORK_DIR}/output_${name}.dat"
    $WIN_ADB pull "$DEVICE_DIR/output_0.dat" "$output_file" > /dev/null 2>&1

    python3 -c "
import numpy as np, json
meta = json.load(open('$meta_path'))
out_q = list(meta['Outputs'].values())[0]['quantize']
qtype = out_q.get('qtype', 'u8')

raw = open('$output_file', 'rb').read()
if qtype == 'i16':
    arr = np.frombuffer(raw, dtype=np.int16)
    fl = out_q['fl']
    if len(arr) == 149*56:
        logits = arr.reshape(149, 56).astype(np.float32) / (2**fl)
    else:
        print(f'  Output size mismatch: {len(arr)} (expected {149*56})')
        exit()
elif qtype == 'i8':
    arr = np.frombuffer(raw, dtype=np.int8)
    scale, zp = out_q['scale'], out_q['zero_point']
    logits = (arr.reshape(149, 56).astype(np.float32) - zp) * scale
else:
    arr = np.frombuffer(raw, dtype=np.uint8)
    scale, zp = out_q['scale'], out_q['zero_point']
    if len(arr) == 56*149:
        logits = (arr.reshape(56,149).T.astype(np.float32) - zp) * scale
    elif len(arr) == 149*56:
        logits = (arr.reshape(149,56).astype(np.float32) - zp) * scale
    else:
        print(f'  Output size mismatch: {len(arr)} (expected {149*56})')
        exit()

tokens = np.argmax(logits, axis=1)
vocab = json.load(open('${WORK_DIR}/vocab.json'))
inv = {v:k for k,v in vocab.items()}
pad_count = np.sum(tokens == 53)
prev = -1
decoded = []
for t in tokens:
    if t != prev and t != 53:
        decoded.append(inv.get(int(t), '?'))
    prev = t
text = ''.join(decoded)
unique, counts = np.unique(tokens, return_counts=True)
top3 = sorted(zip(counts, unique), reverse=True)[:3]
top_str = ', '.join(f'{inv.get(int(t),\"?\")}: {c}' for c,t in top3)
print(f'  PAD: {pad_count}/149, Top: [{top_str}]')
print(f'  Decoded: {text[:80]}')
" 2>&1 | tee -a "$RESULTS"

    echo "" | tee -a "$RESULTS"
done

echo "=== Test Complete ===" | tee -a "$RESULTS"
echo "Results saved to $RESULTS"

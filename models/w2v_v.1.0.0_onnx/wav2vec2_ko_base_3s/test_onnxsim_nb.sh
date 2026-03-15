#!/bin/bash
# Quick test: onnxsim NB vs original NB on T527 NPU
# This tests whether removing dynamic ONNX ops fixes NPU execution
# Run after device recovers from FEL mode

set -e

WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR="/data/local/tmp/w2v_ko_test"

# Check device
echo "Checking device..."
$WIN_ADB devices 2>/dev/null | grep -q "device$" || {
    echo "ERROR: No device found. Is it still in FEL mode?"
    echo "Try: power cycle the device, then run 'usbipd attach --wsl --busid 1-5'"
    exit 1
}

echo "Device found! Setting up test directory..."
$WIN_ADB shell "mkdir -p $DEVICE_DIR" 2>/dev/null

# Test 1: onnxsim NB (MOST IMPORTANT - tests dynamic op removal hypothesis)
echo ""
echo "=== Test 1: onnxsim nopad10_sim_ma (72MB) ==="
echo "This NB has dynamic ONNX ops (Shape/Gather/Concat/Cast) removed via onnxsim."
echo "If this works but original doesn't, it confirms dynamic ops caused NPU failure."

NB1="${WORK_DIR}/wksp/wav2vec2_ko_base_3s_nopad10_sim_ma_nbg_unify/network_binary.nb"
if [ ! -f "$NB1" ]; then
    echo "ERROR: NB not found: $NB1"
    exit 1
fi

# Push NB and input
$WIN_ADB push "$NB1" "$DEVICE_DIR/network_binary.nb"
cp /tmp/input_0_nopad10_sim_ma.dat "${WORK_DIR}/wksp/wav2vec2_ko_base_3s_nopad10_sim_ma_nbg_unify/input_0.dat" 2>/dev/null || true
$WIN_ADB push /tmp/input_0_nopad10_sim_ma.dat "$DEVICE_DIR/input_0.dat"
$WIN_ADB push /tmp/sample_nopad10_sim_ma.txt "$DEVICE_DIR/sample.txt"

# Run NPU inference
echo "Running NPU inference..."
result=$($WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)
echo "$result"

if echo "$result" | grep -q "fail to run\|status=-1"; then
    echo ">>> RESULT: NPU FAILED (onnxsim didn't help)"
else
    echo ">>> RESULT: NPU RAN SUCCESSFULLY!"
    # Pull output and decode
    $WIN_ADB pull "$DEVICE_DIR/output_0.dat" "${WORK_DIR}/output_sim_ma.dat" 2>/dev/null

    python3 -c "
import numpy as np, json
raw = np.fromfile('${WORK_DIR}/output_sim_ma.dat', dtype=np.uint8)
meta = json.load(open('${WORK_DIR}/wksp/wav2vec2_ko_base_3s_nopad10_sim_ma_nbg_unify/nbg_meta.json'))
out_q = list(meta['Outputs'].values())[0]['quantize']
scale, zp = out_q['scale'], out_q['zero_point']
raw = raw.reshape(149, 56)
logits = (raw.astype(np.float32) - zp) * scale
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
print(f'PAD: {pad_count}/149')
print(f'Decoded: {text}')
unique, counts = np.unique(tokens, return_counts=True)
top3 = sorted(zip(counts, unique), reverse=True)[:5]
print(f'Top tokens: {[(inv.get(int(t),\"?\"),c) for c,t in top3]}')
" 2>&1
fi

# Test 2: Original nopad10_ma NB (for comparison)
echo ""
echo "=== Test 2: Original nopad10_ma (72MB, with dynamic ops) ==="
NB2="${WORK_DIR}/wksp/wav2vec2_ko_base_3s_nopad10_ma_nbg_unify/network_binary.nb"
$WIN_ADB push "$NB2" "$DEVICE_DIR/network_binary.nb"
# Use same input but with correct scale
python3 -c "
import numpy as np, json
audio = np.load('${WORK_DIR}/ko_calib_npy/ko_calib_0000.npy').astype(np.float32).flatten()
meta = json.load(open('${WORK_DIR}/wksp/wav2vec2_ko_base_3s_nopad10_ma_nbg_unify/nbg_meta.json'))
inp = list(meta['Inputs'].values())[0]['quantize']
q = np.clip(np.round(audio / inp['scale'] + inp['zero_point']), 0, 255).astype(np.uint8)
q.tofile('/tmp/input_0_orig.dat')
" 2>&1
$WIN_ADB push /tmp/input_0_orig.dat "$DEVICE_DIR/input_0.dat"

result2=$($WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)
echo "$result2"

echo ""
echo "=== Comparison ==="
echo "If Test 1 works and Test 2 doesn't, the dynamic op removal is the key fix."
echo "If both fail, the problem is elsewhere (quantization, weight distribution, etc.)"
echo "If both work, the dynamic ops weren't the issue after all."

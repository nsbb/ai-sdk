#!/bin/bash
# Test all short-input Korean wav2vec2 NBs on T527 NPU
# Run after device is reconnected:
#   powershell.exe "usbipd attach --wsl Ubuntu-20.04 --busid 1-5"
#   sleep 10 && bash test_short_input_nbs.sh

set -e

WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEVICE_DIR="/data/local/tmp/w2v_ko_test"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check device
echo "Checking device..."
$WIN_ADB devices 2>/dev/null | grep -q "device$" || {
    echo "ERROR: No device found."
    echo "Try: powershell.exe 'usbipd attach --wsl Ubuntu-20.04 --busid 1-5'"
    exit 1
}
echo "Device found!"

$WIN_ADB shell "mkdir -p $DEVICE_DIR" 2>/dev/null

# Create test inputs (uint8 quantized)
python3 << 'PYEOF'
import numpy as np

audio = np.load('test_audio.npy').astype(np.float32).flatten()
scale = 0.00427211681380868
zp = 126

# 2s uint8 (speech starts at ~1400ms)
audio_2s = audio[:32000]
q_2s = np.clip(np.round(audio_2s / scale + zp), 0, 255).astype(np.uint8)
q_2s.tofile('/tmp/input_2s_uint8.dat')
print(f"2s uint8: {q_2s.shape}")

# 3s uint8 (full, reference)
audio_3s = audio[:48000]
if len(audio_3s) < 48000:
    audio_3s = np.pad(audio_3s, (0, 48000 - len(audio_3s)))
q_3s = np.clip(np.round(audio_3s / scale + zp), 0, 255).astype(np.uint8)
q_3s.tofile('/tmp/input_3s_uint8.dat')
print(f"3s uint8: {q_3s.shape}")
PYEOF

# Create sample.txt
printf '[network]\n./network_binary.nb\n[input]\n./input_0.dat\n' > /tmp/sample.txt
$WIN_ADB push /tmp/sample.txt "$DEVICE_DIR/sample.txt" 2>&1 | tail -1

echo ""
echo "=========================================="
echo "Test 1: 3s uint8 (reference, should work)"
echo "=========================================="
NB_3S="$WORK_DIR/wksp/wav2vec2_ko_base_3s_nopad10_opset12_sim_ma_nbg_unify/network_binary.nb"
$WIN_ADB push "$NB_3S" "$DEVICE_DIR/network_binary.nb" 2>&1 | tail -1
$WIN_ADB push /tmp/input_3s_uint8.dat "$DEVICE_DIR/input_0.dat" 2>&1 | tail -1
result=$($WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)
echo "$result"
if echo "$result" | grep -q "fail to run\|status=-1"; then
    echo ">>> 3s uint8: FAILED (NPU may still be in bad state)"
else
    echo ">>> 3s uint8: SUCCESS"
    $WIN_ADB pull "$DEVICE_DIR/output_0.dat" "/tmp/output_3s_uint8.dat" 2>/dev/null
fi

echo ""
echo "=========================================="
echo "Test 2: 2s uint8 (shorter input)"
echo "=========================================="
NB_2S="$WORK_DIR/wksp_2s_uint8/wav2vec2_ko_base_2s_uint8_nbg_unify_nbg_unify/network_binary.nb"
$WIN_ADB push "$NB_2S" "$DEVICE_DIR/network_binary.nb" 2>&1 | tail -1
$WIN_ADB push /tmp/input_2s_uint8.dat "$DEVICE_DIR/input_0.dat" 2>&1 | tail -1
result=$($WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)
echo "$result"
if echo "$result" | grep -q "fail to run\|status=-1"; then
    echo ">>> 2s uint8: FAILED"
else
    echo ">>> 2s uint8: SUCCESS"
    $WIN_ADB pull "$DEVICE_DIR/output_0.dat" "/tmp/output_2s_uint8.dat" 2>/dev/null
fi

echo ""
echo "=========================================="
echo "Test 3: 2s int16 DFP"
echo "=========================================="
NB_2S_INT16="$WORK_DIR/wksp_2s_int16/wav2vec2_ko_base_2s_int16_nbg_unify_nbg_unify/network_binary.nb"
python3 -c "
import numpy as np
audio = np.load('test_audio.npy').astype(np.float32).flatten()[:32000]
q = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
q.tofile('/tmp/input_2s_int16.dat')
"
$WIN_ADB push "$NB_2S_INT16" "$DEVICE_DIR/network_binary.nb" 2>&1 | tail -1
$WIN_ADB push /tmp/input_2s_int16.dat "$DEVICE_DIR/input_0.dat" 2>&1 | tail -1
result=$($WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)
echo "$result"
if echo "$result" | grep -q "fail to run\|status=-1"; then
    echo ">>> 2s int16: FAILED (expected)"
else
    echo ">>> 2s int16: SUCCESS (unexpected!)"
    $WIN_ADB pull "$DEVICE_DIR/output_0.dat" "/tmp/output_2s_int16.dat" 2>/dev/null
fi

echo ""
echo "=========================================="
echo "Decoding outputs"
echo "=========================================="
python3 << 'DECODE'
import numpy as np, json, os

vocab = json.load(open('vocab.json'))
inv = {v: k for k, v in vocab.items()}

def decode_output(path, seq_len, scale, zp, label):
    if not os.path.exists(path):
        print(f"  [{label}] No output file")
        return
    raw = np.fromfile(path, dtype=np.uint8)
    try:
        logits = (raw.reshape(seq_len, 56).astype(np.float32) - zp) * scale
    except:
        print(f"  [{label}] Shape mismatch: {raw.shape}")
        return

    tokens = np.argmax(logits, axis=1)
    pad_count = np.sum(tokens == 53)

    prev = -1
    decoded = []
    for t in tokens:
        if t != prev:
            if t != 53:  # [PAD] = blank
                decoded.append(inv.get(int(t), '?'))
            prev = t
    text = ''.join(decoded)

    unique, counts = np.unique(tokens, return_counts=True)
    top3 = sorted(zip(counts, unique), reverse=True)[:5]
    top3_str = [(inv.get(int(t),'?'), c) for c, t in top3]

    print(f"  [{label}] PAD: {pad_count}/{seq_len}")
    print(f"  [{label}] Decoded: '{text}'")
    print(f"  [{label}] Top tokens: {top3_str}")

# output scale/zp from quantize file
out_scale = 0.0873851403594017
out_zp = 117

decode_output('/tmp/output_3s_uint8.dat', 149, out_scale, out_zp, '3s uint8')
decode_output('/tmp/output_2s_uint8.dat', 99, out_scale, out_zp, '2s uint8')
DECODE

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "int16 DFP: NPU status=-1 regardless of input length (1s/2s/3s)"
echo "uint8: Check above for results"
echo "1s model: Too short for Korean speech recognition (all PAD in FP32)"

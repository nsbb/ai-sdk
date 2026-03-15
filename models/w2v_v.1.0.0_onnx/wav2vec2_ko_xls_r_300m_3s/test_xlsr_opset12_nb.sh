#!/bin/bash
# Test XLS-R-300M 12-layer opset12+sim NB on T527 NPU
# This NB was re-exported with opset 12 (manual attention) to remove dynamic ops

set -e
WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR="/data/local/tmp/xlsr_ko_test"
AUDIO_NPY="${WORK_DIR}/../wav2vec2_ko_base_3s/ko_calib_npy/ko_calib_0000.npy"

echo "=== XLS-R 12L Opset12+Sim NPU Test ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"

# Check device
$WIN_ADB devices | grep -q "device$" || { echo "ERROR: No device found"; exit 1; }
$WIN_ADB shell "mkdir -p $DEVICE_DIR" 2>/dev/null

NB_DIR="${WORK_DIR}/wksp/wav2vec2_ko_xlsr_12L_nopad10_opset12_sim_ma_nbg_unify"
nb_path="${NB_DIR}/network_binary.nb"
meta_path="${NB_DIR}/nbg_meta.json"

if [ ! -f "$nb_path" ]; then
    echo "ERROR: NB not found: $nb_path"
    exit 1
fi

nb_size=$(ls -lh "$nb_path" | awk '{print $5}')
echo "NB: $nb_size"

# Generate quantized input
python3 -c "
import numpy as np, json, sys
audio = np.load('$AUDIO_NPY').astype(np.float32).flatten()[:48000]
if len(audio) < 48000: audio = np.pad(audio, (0, 48000-len(audio)))
meta = json.load(open('$meta_path'))
inp = list(meta['Inputs'].values())[0]['quantize']
scale, zp = inp['scale'], inp['zero_point']
q = np.clip(np.round(audio / scale + zp), 0, 255).astype(np.uint8)
q.tofile('/tmp/xlsr_input_0.dat')
print(f'Input: scale={scale:.6f} zp={zp}', file=sys.stderr)
" 2>&1

# Push and run
$WIN_ADB push "$nb_path" "$DEVICE_DIR/network_binary.nb" > /dev/null 2>&1
$WIN_ADB push /tmp/xlsr_input_0.dat "$DEVICE_DIR/input_0.dat" > /dev/null 2>&1
$WIN_ADB shell "echo 'input_0.dat' > $DEVICE_DIR/sample.txt" 2>/dev/null
sleep 1

result=$(timeout 60 $WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)

if echo "$result" | grep -q "fail to run\|status=-1"; then
    echo ">>> NPU FAILED"
    echo "$result"
    exit 1
fi

inference_time=$(echo "$result" | grep "run time" | grep -oP '\d+(?= us)')
echo "Inference: ${inference_time}us"

# Pull and decode output
$WIN_ADB pull "$DEVICE_DIR/output_0.dat" "${WORK_DIR}/output_xlsr_12L_opset12.dat" > /dev/null 2>&1

python3 -c "
import numpy as np, json

meta = json.load(open('$meta_path'))
out_q = list(meta['Outputs'].values())[0]['quantize']
scale, zp = out_q['scale'], out_q['zero_point']

raw = np.fromfile('${WORK_DIR}/output_xlsr_12L_opset12.dat', dtype=np.uint8)
expected = 149 * 1205
if len(raw) == expected:
    logits = (raw.reshape(149, 1205).astype(np.float32) - zp) * scale
elif len(raw) == 1205 * 149:
    logits = (raw.reshape(1205, 149).T.astype(np.float32) - zp) * scale
else:
    print(f'Output size mismatch: {len(raw)} (expected {expected})')
    exit()

tokens = np.argmax(logits, axis=1)
vocab = json.load(open('${WORK_DIR}/vocab_xlsr.json'))
inv = {v: k for k, v in vocab.items()}
blank_id = 1204

pad_count = np.sum(tokens == blank_id)
prev = -1
decoded = []
for t in tokens:
    if t != prev and t != blank_id:
        decoded.append(inv.get(int(t), '?'))
    prev = t
text = ''.join(decoded).replace('|', ' ')

unique, counts = np.unique(tokens, return_counts=True)
top3 = sorted(zip(counts, unique), reverse=True)[:5]
top_str = ', '.join(f'{inv.get(int(t),\"?\")}: {c}' for c, t in top3)
print(f'PAD: {pad_count}/149, Top: [{top_str}]')
print(f'Decoded: {text[:100]}')
" 2>&1

echo ""
echo "=== Test Complete ==="

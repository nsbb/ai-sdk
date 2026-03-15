#!/bin/bash
# Test split wav2vec2 model: Part A (CNN uint8) → Part B (Transformer int16)
# This avoids uint8 quantization degradation in the transformer layers

set -e
WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR="/data/local/tmp/w2v_ko_split"
AUDIO_NPY="${WORK_DIR}/ko_calib_npy/ko_calib_0000.npy"

echo "=== Split Model Test: PartA(uint8 CNN) → PartB(int16 Transformer) ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"

# Check device
$WIN_ADB devices | grep -q "device$" || { echo "ERROR: No device found"; exit 1; }
$WIN_ADB shell "mkdir -p $DEVICE_DIR" 2>/dev/null

# === Part A: CNN Feature Extractor (uint8, 3.7MB) ===
echo ""
echo "--- Part A: CNN Feature Extractor (uint8, 3.7MB) ---"

PART_A_NB="${WORK_DIR}/wksp/partA_uint8_nbg_unify_nbg_unify/network_binary.nb"
PART_A_META="${WORK_DIR}/wksp/partA_uint8_nbg_unify_nbg_unify/nbg_meta.json"

if [ ! -f "$PART_A_NB" ]; then
    echo "ERROR: Part A NB not found"
    exit 1
fi

# Generate quantized input for Part A (audio → uint8)
python3 -c "
import numpy as np, json
audio = np.load('$AUDIO_NPY').astype(np.float32).flatten()[:48000]
if len(audio) < 48000: audio = np.pad(audio, (0, 48000-len(audio)))
meta = json.load(open('$PART_A_META'))
inp = list(meta['Inputs'].values())[0]['quantize']
scale, zp = inp['scale'], inp['zero_point']
q = np.clip(np.round(audio / scale + zp), 0, 255).astype(np.uint8)
q.tofile('/tmp/partA_input.dat')
print(f'  PartA Input: u8 scale={scale:.6f} zp={zp}')
" 2>&1

# Push and run Part A
$WIN_ADB push "$PART_A_NB" "$DEVICE_DIR/network_binary.nb" > /dev/null 2>&1
$WIN_ADB push /tmp/partA_input.dat "$DEVICE_DIR/input_0.dat" > /dev/null 2>&1
$WIN_ADB shell "echo 'input_0.dat' > $DEVICE_DIR/sample.txt" 2>/dev/null
sleep 1

result_a=$(timeout 30 $WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)

if echo "$result_a" | grep -q "fail to run\|status=-1"; then
    echo "  >>> Part A FAILED"
    echo "$result_a"
    exit 1
fi

time_a=$(echo "$result_a" | grep "run time" | grep -oP '\d+(?= us)')
echo "  Part A inference: ${time_a}us"

# Pull Part A output
$WIN_ADB pull "$DEVICE_DIR/output_0.dat" "/tmp/partA_output.dat" > /dev/null 2>&1

# === Convert Part A output → Part B input ===
echo ""
echo "--- Converting PartA output (u8) → PartB input (i16) ---"

PART_B_NB_I16="${WORK_DIR}/wksp/partB_int16_nbg_unify_nbg_unify/network_binary.nb"
PART_B_NB_U8="${WORK_DIR}/wksp/partB_uint8_nbg_unify_nbg_unify/network_binary.nb"
PART_B_META_I16="${WORK_DIR}/wksp/partB_int16_nbg_unify_nbg_unify/nbg_meta.json"
PART_B_META_U8="${WORK_DIR}/wksp/partB_uint8_nbg_unify_nbg_unify/nbg_meta.json"

# Test both Part B variants
for variant in "int16" "uint8"; do
    if [ "$variant" = "int16" ]; then
        PART_B_NB="$PART_B_NB_I16"
        PART_B_META="$PART_B_META_I16"
    else
        PART_B_NB="$PART_B_NB_U8"
        PART_B_META="$PART_B_META_U8"
    fi

    if [ ! -f "$PART_B_NB" ]; then
        echo "  SKIP: Part B $variant NB not found"
        continue
    fi

    echo ""
    echo "--- Part B: Transformer ($variant) ---"
    nb_size=$(ls -lh "$PART_B_NB" | awk '{print $5}')
    echo "  NB size: $nb_size"

    # Convert Part A u8 output → Part B input
    python3 -c "
import numpy as np, json

# Read Part A output (u8, shape [1,149,768])
partA_meta = json.load(open('$PART_A_META'))
out_q = list(partA_meta['Outputs'].values())[0]['quantize']
partA_raw = np.fromfile('/tmp/partA_output.dat', dtype=np.uint8)
print(f'  PartA output size: {len(partA_raw)} (expected {1*149*768})')

# Dequantize Part A output
partA_float = (partA_raw.astype(np.float32) - out_q['zero_point']) * out_q['scale']

# Quantize to Part B input format
partB_meta = json.load(open('$PART_B_META'))
inp_q = list(partB_meta['Inputs'].values())[0]['quantize']
qtype = inp_q.get('qtype', 'u8')

if qtype == 'i16':
    fl = inp_q['fl']
    q = np.clip(np.round(partA_float * (2**fl)), -32768, 32767).astype(np.int16)
    q.tofile('/tmp/partB_input.dat')
    print(f'  PartB Input: i16 fl={fl}, range [{q.min()}, {q.max()}]')
else:
    scale, zp = inp_q['scale'], inp_q['zero_point']
    q = np.clip(np.round(partA_float / scale + zp), 0, 255).astype(np.uint8)
    q.tofile('/tmp/partB_input.dat')
    print(f'  PartB Input: u8 scale={scale:.6f} zp={zp}')
" 2>&1

    # Push and run Part B
    $WIN_ADB push "$PART_B_NB" "$DEVICE_DIR/network_binary.nb" > /dev/null 2>&1
    $WIN_ADB push /tmp/partB_input.dat "$DEVICE_DIR/input_0.dat" > /dev/null 2>&1
    $WIN_ADB shell "echo 'input_0.dat' > $DEVICE_DIR/sample.txt" 2>/dev/null
    sleep 1

    result_b=$(timeout 60 $WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)

    if echo "$result_b" | grep -q "fail to run\|status=-1"; then
        echo "  >>> Part B ($variant) FAILED"
        continue
    fi

    time_b=$(echo "$result_b" | grep "run time" | grep -oP '\d+(?= us)')
    total_time=$((time_a + time_b))
    echo "  Part B inference: ${time_b}us"
    echo "  Total (A+B): ${total_time}us"

    # Pull and decode output
    $WIN_ADB pull "$DEVICE_DIR/output_0.dat" "/tmp/partB_output_${variant}.dat" > /dev/null 2>&1

    python3 -c "
import numpy as np, json

meta = json.load(open('$PART_B_META'))
out_q = list(meta['Outputs'].values())[0]['quantize']
qtype = out_q.get('qtype', 'u8')

raw = open('/tmp/partB_output_${variant}.dat', 'rb').read()

if qtype == 'i16':
    arr = np.frombuffer(raw, dtype=np.int16)
    fl = out_q['fl']
    if len(arr) == 149*56:
        logits = arr.reshape(149, 56).astype(np.float32) / (2**fl)
    else:
        print(f'  Output size mismatch: {len(arr)} (expected {149*56})')
        exit()
else:
    arr = np.frombuffer(raw, dtype=np.uint8)
    scale, zp = out_q['scale'], out_q['zero_point']
    if len(arr) == 149*56:
        logits = (arr.reshape(149, 56).astype(np.float32) - zp) * scale
    elif len(arr) == 56*149:
        logits = (arr.reshape(56, 149).T.astype(np.float32) - zp) * scale
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
top_str = ', '.join(f'{inv.get(int(t),chr(63))}: {c}' for c,t in top3)
print(f'  PAD: {pad_count}/149, Top: [{top_str}]')
print(f'  Decoded: {text[:80]}')
" 2>&1

done

echo ""
echo "=== Split Model Test Complete ==="

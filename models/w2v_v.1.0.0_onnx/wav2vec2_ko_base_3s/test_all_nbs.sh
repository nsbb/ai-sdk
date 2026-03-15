#!/bin/bash
# Test all Korean wav2vec2 NB variants on T527 NPU
# Run after device is connected

WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
WORK_DIR="/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s"
DEVICE_DIR="/data/local/tmp/w2v_ko_test"
AUDIO_NPY="${WORK_DIR}/ko_calib_npy/ko_calib_0000.npy"

# NB variants to test
declare -A NBS=(
    ["nopad10_ma"]="wksp/wav2vec2_ko_base_3s_nopad10_ma_nbg_unify/network_binary.nb"
    ["temp3_nopad10_ma"]="wksp/wav2vec2_ko_base_3s_temp3_nopad10_ma_nbg_unify/network_binary.nb"
    ["temp5_nopad10_ma"]="wksp/wav2vec2_ko_base_3s_temp5_nopad10_ma_nbg_unify/network_binary.nb"
    ["nopad10_relu_ma"]="wksp/wav2vec2_ko_base_3s_nopad10_relu_ma_nbg_unify/network_binary.nb"
    ["nopad10_6layers_ma"]="wksp/wav2vec2_ko_base_3s_nopad10_6layers_ma_nbg_unify/network_binary.nb"
    ["nopad10_hybrid_ma"]="wksp/wav2vec2_ko_base_3s_nopad10_hybrid_ma_nbg_unify/network_binary.nb"
    ["cnn_only_ma"]="wksp/wav2vec2_ko_base_3s_cnn_only_ma_nbg_unify/network_binary.nb"
    ["nopad10_sim_ma"]="wksp/wav2vec2_ko_base_3s_nopad10_sim_ma_nbg_unify/network_binary.nb"
    ["combo_relu_6l_nopad10_ma"]="wksp/wav2vec2_ko_base_3s_combo_relu_6l_nopad10_ma_nbg_unify/network_binary.nb"
    ["nopad10_normal"]="wksp/wav2vec2_ko_base_3s_nopad10_normal_nbg_unify/network_binary.nb"
    ["nopad10_auto"]="wksp/wav2vec2_ko_base_3s_nopad10_auto_nbg_unify/network_binary.nb"
    ["nopad10_opset12_ma"]="wksp/wav2vec2_ko_base_3s_nopad10_opset12_ma_nbg_unify/network_binary.nb"
    ["nopad10_opset12_sim_ma"]="wksp/wav2vec2_ko_base_3s_nopad10_opset12_sim_ma_nbg_unify/network_binary.nb"
    ["6L_nopad10_opset12_sim_ma"]="wksp/wav2vec2_ko_base_3s_6L_nopad10_opset12_sim_ma_nbg_unify/network_binary.nb"
    ["relu_nopad10_opset12_sim_ma"]="wksp/wav2vec2_ko_base_3s_relu_nopad10_opset12_sim_ma_nbg_unify/network_binary.nb"
    ["6L_relu_nopad10_opset12_sim_ma"]="wksp/wav2vec2_ko_base_3s_6L_relu_nopad10_opset12_sim_ma_nbg_unify/network_binary.nb"
    ["combo_sim_relu_6l_nopad10_ma"]="wksp/wav2vec2_ko_base_3s_combo_sim_relu_6l_nopad10_ma_nbg_unify/network_binary.nb"
    ["nopad10_opset12_sim_normal"]="wksp/wav2vec2_ko_base_3s_nopad10_opset12_sim_normal_nbg_unify/network_binary.nb"
    ["nopad10_opset12_sim_kl"]="wksp/wav2vec2_ko_base_3s_nopad10_opset12_sim_kl_nbg_unify/network_binary.nb"
    ["nopad10_opset12_sim_hybrid_ma"]="wksp/wav2vec2_ko_base_3s_nopad10_opset12_sim_hybrid_ma_nbg_unify/network_binary.nb"
    ["nopad10_opset12_sim_int16"]="wksp/wav2vec2_ko_base_3s_nopad10_opset12_sim_int16_nbg_unify/network_binary.nb"
    ["nopad10_opset12_sim_int8"]="wksp/wav2vec2_ko_base_3s_nopad10_opset12_sim_int8_nbg_unify/network_binary.nb"
)

echo "=== Korean wav2vec2 NB Batch Test ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check device
$WIN_ADB devices | grep -q "device$" || { echo "ERROR: No device found"; exit 1; }

$WIN_ADB shell "mkdir -p $DEVICE_DIR" 2>/dev/null

for name in "${!NBS[@]}"; do
    nb_path="${WORK_DIR}/${NBS[$name]}"
    if [ ! -f "$nb_path" ]; then
        echo "SKIP: $name (NB not found)"
        continue
    fi

    echo "--- Testing: $name ---"
    nb_size=$(stat -c%s "$nb_path" 2>/dev/null)
    echo "  NB size: $(echo "$nb_size / 1048576" | bc)MB"

    # Generate per-variant quantized input
    meta_path="${nb_path/network_binary.nb/nbg_meta.json}"
    python3 -c "
import numpy as np, json
audio = np.load('$AUDIO_NPY').astype(np.float32).flatten()[:48000]
if len(audio) < 48000: audio = np.pad(audio, (0, 48000-len(audio)))
meta = json.load(open('$meta_path'))
inp = list(meta['Inputs'].values())[0]['quantize']
q = np.clip(np.round(audio / inp['scale'] + inp['zero_point']), 0, 255).astype(np.uint8)
q.tofile('/tmp/input_0_test.dat')
" 2>&1

    # Push NB and input
    $WIN_ADB push "$nb_path" "$DEVICE_DIR/network_binary.nb" > /dev/null 2>&1
    $WIN_ADB push /tmp/input_0_test.dat "$DEVICE_DIR/input_0.dat" > /dev/null 2>&1
    sleep 1

    # Create sample.txt if needed
    $WIN_ADB shell "echo 'input_0.dat' > $DEVICE_DIR/sample.txt" 2>/dev/null

    # Run vpm_run
    result=$(timeout 30 $WIN_ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -b 0" 2>&1)
    
    if echo "$result" | grep -q "fail to run"; then
        echo "  Result: FAILED (NPU error)"
        continue
    fi
    
    inference_time=$(echo "$result" | grep "run time" | grep -oP '\d+(?= us)')
    echo "  Inference: ${inference_time}us"
    
    # Pull output
    output_file="${WORK_DIR}/output_${name}.dat"
    $WIN_ADB pull "$DEVICE_DIR/output_0.dat" "$output_file" > /dev/null 2>&1
    
    # Decode with Python
    python3 -c "
import numpy as np, json
raw = np.fromfile('$output_file', dtype=np.uint8)
meta_path = '${WORK_DIR}/${NBS[$name]}'.replace('network_binary.nb', 'nbg_meta.json')
try:
    meta = json.load(open(meta_path))
    out_q = list(meta['Outputs'].values())[0]['quantize']
    scale, zp = out_q['scale'], out_q['zero_point']
except:
    scale, zp = 0.084869, 117

vocab = json.load(open('${WORK_DIR}/vocab.json'))
inv = {v:k for k,v in vocab.items()}

if len(raw) == 56*149:
    raw = raw.reshape(56,149).T
elif len(raw) == 149*56:
    raw = raw.reshape(149,56)
logits = (raw.astype(np.float32) - zp) * scale
tokens = np.argmax(logits, axis=1)
pad_count = np.sum(tokens == 53)

prev = -1
decoded = []
for t in tokens:
    if t != prev and t != 53:
        decoded.append(inv.get(int(t), '?'))
    prev = t
text = ''.join(decoded)
unique, counts = np.unique(tokens, return_counts=True)
top_token = unique[np.argmax(counts)]
top_count = counts[np.argmax(counts)]
print(f'  PAD: {pad_count}/149, Top token: {top_token}({top_count}), Decoded: {text[:60]}')
" 2>&1
    
    echo ""
done

echo "=== Test Complete ==="

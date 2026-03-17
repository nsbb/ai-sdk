#!/bin/bash
# Pegasus 양자화 파이프라인: clip50 모델
# 실행 환경: WSL에서 source env.sh v2 후 실행
#
# 사전 조건:
#   source /home/nsbb/travail/claude/T527/ai-sdk/models/env.sh v2
#   export ACUITY_PATH=/nas02/geonhui83/T527_toolkit/acuity-toolkit-binary-6.12.0/bin/
#   export VIV_SDK=/home1/Gunhee_Lee/VeriSilicon/VivanteIDE5.7.2/cmdtools

set -e

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"
ONNX_FILE="wav2vec2_ko_base_3s_clip50_nopad10_opset12_sim.onnx"
MODEL_NAME="wav2vec2_ko_base_3s_clip50"
WKSP="${MODEL_DIR}/wksp_clip50"

echo "=== Pegasus Pipeline: ${MODEL_NAME} ==="
echo "Model dir: ${MODEL_DIR}"
echo "ONNX: ${ONNX_FILE}"

# Check environment
if [ -z "$ACUITY_PATH" ]; then
    echo "ERROR: ACUITY_PATH not set. Run: source env.sh v2"
    exit 1
fi

mkdir -p "${WKSP}"
cd "${WKSP}"

# --- Step 1: Import ---
echo ""
echo "=== Step 1: Import ONNX ==="
pegasus import onnx \
    --model "${MODEL_DIR}/${ONNX_FILE}" \
    --output-model "${MODEL_NAME}.json" \
    --output-data "${MODEL_NAME}.data" \
    --inputs "input_values" \
    --input-size-list "1,48000"

# --- Step 2: Input meta ---
echo ""
echo "=== Step 2: Create inputmeta ==="
cat > "${MODEL_NAME}_inputmeta.yml" << 'INPUTMETA'
---
input_0:
  source_format: "raw"
  mean: [0.0]
  scale: [1.0]
  reverse_channel: false
INPUTMETA

# --- Step 3: Prepare calibration data ---
echo ""
echo "=== Step 3: Prepare calibration dataset ==="

# calibration npy를 dataset.txt로 변환
CALIB_DIR="${MODEL_DIR}/aug_calib_npy"
CALIB_TENSOR_DIR="${WKSP}/calib_tensors"
mkdir -p "${CALIB_TENSOR_DIR}"

python3 << 'PYTHON_CALIB'
import os, glob, numpy as np

calib_dir = os.environ.get('CALIB_DIR', 'aug_calib_npy')
tensor_dir = os.environ.get('CALIB_TENSOR_DIR', 'calib_tensors')
wksp = os.environ.get('WKSP', '.')

files = sorted(glob.glob(os.path.join(calib_dir, '*.npy')))[:50]
dataset_lines = []

for i, f in enumerate(files):
    data = np.load(f).astype(np.float32)
    if data.shape != (1, 48000):
        data = data.reshape(1, -1)[:, :48000]
        if data.shape[1] < 48000:
            data = np.pad(data, ((0, 0), (0, 48000 - data.shape[1])))

    out_path = os.path.join(tensor_dir, f'calib_{i:04d}.tensor')
    data.tofile(out_path)
    dataset_lines.append(out_path)

with open(os.path.join(wksp, 'dataset.txt'), 'w') as f:
    for line in dataset_lines:
        f.write(line + '\n')

print(f"Created {len(dataset_lines)} calibration tensors")
PYTHON_CALIB

export CALIB_DIR CALIB_TENSOR_DIR WKSP

# --- Step 4: Quantize ---
echo ""
echo "=== Step 4: Quantize (uint8, moving_average) ==="
pegasus quantize \
    --model "${MODEL_NAME}.json" \
    --model-data "${MODEL_NAME}.data" \
    --with-input-meta "${MODEL_NAME}_inputmeta.yml" \
    --device CPU \
    --quantizer asymmetric_affine --qtype uint8 \
    --rebuild-all \
    --algorithm moving_average --moving-average-weight 0.004 \
    --model-quantize "${MODEL_NAME}_uint8.quantize"

# --- Step 5: Inference simulation ---
echo ""
echo "=== Step 5: Inference simulation ==="

# 테스트 입력 준비
python3 << 'PYTHON_TEST'
import numpy as np, os

test = np.load(os.path.join(os.environ['MODEL_DIR'], 'test_audio.npy')).astype(np.float32)
test.tofile('test_input.tensor')
print(f"Test input: shape={test.shape}, range=[{test.min():.4f}, {test.max():.4f}]")
PYTHON_TEST

export MODEL_DIR

# FP32 시뮬레이션
pegasus inference \
    --model "${MODEL_NAME}.json" \
    --model-data "${MODEL_NAME}.data" \
    --device CPU --dtype float \
    --with-input-meta "${MODEL_NAME}_inputmeta.yml" \
    --input-file "test_input.tensor" \
    --output-path "${WKSP}/output_fp32/"

# uint8 시뮬레이션
pegasus inference \
    --model "${MODEL_NAME}.json" \
    --model-data "${MODEL_NAME}.data" \
    --model-quantize "${MODEL_NAME}_uint8.quantize" \
    --device CPU --dtype quantized \
    --with-input-meta "${MODEL_NAME}_inputmeta.yml" \
    --input-file "test_input.tensor" \
    --output-path "${WKSP}/output_uint8/"

# 결과 비교
echo ""
echo "=== Comparing FP32 vs uint8 simulation ==="
python3 << 'PYTHON_COMPARE'
import numpy as np, os, json

# output 파일 찾기
fp32_dir = os.path.join(os.environ['WKSP'], 'output_fp32')
uint8_dir = os.path.join(os.environ['WKSP'], 'output_uint8')

fp32_files = sorted([f for f in os.listdir(fp32_dir) if f.endswith('.tensor')])
uint8_files = sorted([f for f in os.listdir(uint8_dir) if f.endswith('.tensor')])

if fp32_files and uint8_files:
    fp32 = np.fromfile(os.path.join(fp32_dir, fp32_files[0]), dtype=np.float32).reshape(1, 149, 56)
    uint8 = np.fromfile(os.path.join(uint8_dir, uint8_files[0]), dtype=np.float32).reshape(1, 149, 56)

    print(f"FP32: range=[{fp32.min():.4f}, {fp32.max():.4f}]")
    print(f"uint8: range=[{uint8.min():.4f}, {uint8.max():.4f}]")

    # argmax agreement
    fp32_tokens = np.argmax(fp32[0], axis=-1)
    uint8_tokens = np.argmax(uint8[0], axis=-1)
    agree = np.mean(fp32_tokens == uint8_tokens)
    print(f"Argmax agreement: {agree*100:.1f}%")

    # CTC decode
    with open(os.path.join(os.environ['MODEL_DIR'], 'vocab.json')) as f:
        vd = json.load(f)
    vocab = [''] * (max(vd.values()) + 1)
    for c, i in vd.items(): vocab[i] = c

    def ctc(logits):
        tokens = np.argmax(logits, axis=-1)
        prev=-1; dec=[]
        for t in tokens:
            if t!=prev:
                if t!=0: dec.append(t)
                prev=t
        return ''.join(vocab[t] for t in dec if t<len(vocab))

    print(f"FP32 decoded: {ctc(fp32[0])}")
    print(f"uint8 decoded: {ctc(uint8[0])}")
else:
    print("No output files found")
    print(f"FP32 dir: {os.listdir(fp32_dir)}")
    print(f"uint8 dir: {os.listdir(uint8_dir)}")
PYTHON_COMPARE

# --- Step 6: Export NB (only if simulation looks good) ---
echo ""
echo "=== Step 6: Export NB ==="
echo "Run manually after checking simulation results:"
echo ""
echo "pegasus export ovxlib \\"
echo "  --model ${MODEL_NAME}.json \\"
echo "  --model-data ${MODEL_NAME}.data \\"
echo "  --model-quantize ${MODEL_NAME}_uint8.quantize \\"
echo "  --dtype quantized \\"
echo "  --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016 \\"
echo "  --viv-sdk \$VIV_SDK/vsimulator --target-ide-project linux64 --batch-size 1 \\"
echo "  --output-path ${WKSP}/${MODEL_NAME}_uint8_nbg_unify/"

echo ""
echo "=== Pipeline complete ==="

#!/bin/bash
# Pegasus pipeline: import → quantize uint8 → inference (시뮬) → 비교
# eager opset12 re-export된 한국어 wav2vec2로 양자화 테스트

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="wav2vec2_ko_eager_op12_3s"
ONNX_FILE="${SCRIPT_DIR}/${MODEL_NAME}.onnx"
CALIB_NPY_DIR="${SCRIPT_DIR}/../wav2vec2_ko_xls_r_300m_3s/calib_data_v2"

# Acuity 환경
ACUITY_PATH="/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0/bin"
export PATH="${ACUITY_PATH}:${PATH}"

WKSP="${SCRIPT_DIR}/wksp"
mkdir -p "${WKSP}"
cd "${WKSP}"

# --- Step 1: Import ---
echo "=== Step 1: Pegasus Import ==="
if [ ! -f "${MODEL_NAME}.json" ]; then
    pegasus import onnx \
        --model "${ONNX_FILE}" \
        --output-model "${MODEL_NAME}.json" \
        --output-data "${MODEL_NAME}.data"
else
    echo "Skipping (already imported)"
fi

# --- Step 2: Prepare calibration data ---
echo ""
echo "=== Step 2: Prepare calibration data (npy → tensor) ==="
CALIB_TENSOR_DIR="${WKSP}/calib_tensors"
mkdir -p "${CALIB_TENSOR_DIR}"

export CALIB_NPY_DIR CALIB_TENSOR_DIR WKSP
python3 << 'PYTHON_CALIB'
import os, glob, numpy as np

calib_dir = os.environ['CALIB_NPY_DIR']
tensor_dir = os.environ['CALIB_TENSOR_DIR']
wksp = os.environ['WKSP']

files = sorted(glob.glob(os.path.join(calib_dir, '*.npy')))[:50]
dataset_lines = []

for i, f in enumerate(files):
    data = np.load(f).astype(np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
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
if dataset_lines:
    d = np.load(files[0]).astype(np.float32).reshape(1, -1)[:, :48000]
    print(f"  Sample shape: {d.shape}, range: [{d.min():.4f}, {d.max():.4f}]")
PYTHON_CALIB

# --- Step 3: Input meta (Acuity 6.12 format) ---
echo ""
echo "=== Step 3: Create inputmeta ==="
cat > "${MODEL_NAME}_inputmeta.yml" << 'INPUTMETA'
# !!!This file disallow TABs!!!
input_meta:
  databases:
  - path: dataset.txt
    type: TEXT
    ports:
    - lid: input_values_604
      category: frequency
      dtype: float32
      sparse: false
      tensor_name:
      shape:
      - 1
      - 48000
      fitting: scale
      preprocess:
        reverse_channel: false
        scale: 1.0
        preproc_node_params:
          add_preproc_node: false
          preproc_type: TENSOR
          preproc_perm:
          - 0
          - 1
      redirect_to_output: false
INPUTMETA

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

# --- Step 5: Prepare test input ---
echo ""
echo "=== Step 5: Prepare test input ==="
python3 << 'PYTHON_TEST'
import numpy as np, os, glob

calib_dir = os.environ['CALIB_NPY_DIR']
files = sorted(glob.glob(os.path.join(calib_dir, '*.npy')))
if files:
    test = np.load(files[0]).astype(np.float32)
    if test.ndim == 1:
        test = test.reshape(1, -1)
    test = test[:, :48000]
    if test.shape[1] < 48000:
        test = np.pad(test, ((0, 0), (0, 48000 - test.shape[1])))
    test.tofile('test_input.tensor')
    print(f"Test input: shape={test.shape}, range=[{test.min():.4f}, {test.max():.4f}]")
PYTHON_TEST

# test input을 위한 dataset_test.txt
echo "${WKSP}/test_input.tensor" > dataset_test.txt

# test용 inputmeta (dataset_test.txt 참조)
cat > "${MODEL_NAME}_inputmeta_test.yml" << 'INPUTMETA_TEST'
# !!!This file disallow TABs!!!
input_meta:
  databases:
  - path: dataset_test.txt
    type: TEXT
    ports:
    - lid: input_values_604
      category: frequency
      dtype: float32
      sparse: false
      tensor_name:
      shape:
      - 1
      - 48000
      fitting: scale
      preprocess:
        reverse_channel: false
        scale: 1.0
        preproc_node_params:
          add_preproc_node: false
          preproc_type: TENSOR
          preproc_perm:
          - 0
          - 1
      redirect_to_output: false
INPUTMETA_TEST

# --- Step 6: FP32 inference ---
echo ""
echo "=== Step 6: Inference (FP32 golden) ==="
mkdir -p output_fp32
pegasus inference \
    --model "${MODEL_NAME}.json" \
    --model-data "${MODEL_NAME}.data" \
    --device CPU --dtype float \
    --with-input-meta "${MODEL_NAME}_inputmeta_test.yml" \
    --output-path "${WKSP}/output_fp32/"

# --- Step 7: uint8 inference ---
echo ""
echo "=== Step 7: Inference (uint8 simulation) ==="
mkdir -p output_uint8
pegasus inference \
    --model "${MODEL_NAME}.json" \
    --model-data "${MODEL_NAME}.data" \
    --model-quantize "${MODEL_NAME}_uint8.quantize" \
    --device CPU --dtype quantized \
    --with-input-meta "${MODEL_NAME}_inputmeta_test.yml" \
    --output-path "${WKSP}/output_uint8/"

# --- Step 8: Compare ---
echo ""
echo "=== Step 8: Compare FP32 vs uint8 ==="
export SCRIPT_DIR WKSP
python3 << 'PYTHON_COMPARE'
import numpy as np, os, json

wksp = os.environ['WKSP']
script_dir = os.environ['SCRIPT_DIR']
fp32_dir = os.path.join(wksp, 'output_fp32')
uint8_dir = os.path.join(wksp, 'output_uint8')

fp32_files = sorted([f for f in os.listdir(fp32_dir) if f.endswith('.tensor')])
uint8_files = sorted([f for f in os.listdir(uint8_dir) if f.endswith('.tensor')])

if fp32_files and uint8_files:
    # Output shape: [1, 149, 56] for 3s model
    fp32 = np.fromfile(os.path.join(fp32_dir, fp32_files[0]), dtype=np.float32).reshape(1, 149, 56)
    uint8 = np.fromfile(os.path.join(uint8_dir, uint8_files[0]), dtype=np.float32).reshape(1, 149, 56)

    print(f"FP32:  range=[{fp32.min():.4f}, {fp32.max():.4f}]")
    print(f"uint8: range=[{uint8.min():.4f}, {uint8.max():.4f}]")

    # Argmax agreement
    fp32_tokens = np.argmax(fp32[0], axis=-1)
    uint8_tokens = np.argmax(uint8[0], axis=-1)
    agree = np.mean(fp32_tokens == uint8_tokens) * 100
    print(f"\nArgmax agreement: {agree:.1f}%")
    print(f"  (old KO model: 46.3% — this should be higher)")

    # CTC decode
    vocab_path = os.path.join(script_dir, '..', 'wav2vec2_ko_base_3s', 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vd = json.load(f)
        vocab = [''] * (max(vd.values()) + 1)
        for c, i in vd.items():
            vocab[i] = c

        def ctc(logits):
            tokens = np.argmax(logits, axis=-1)
            prev = -1
            dec = []
            for t in tokens:
                if t != prev:
                    if t != 0:
                        dec.append(t)
                    prev = t
            return ''.join(vocab[t] for t in dec if t < len(vocab))

        print(f"\nFP32  decoded: {ctc(fp32[0])}")
        print(f"uint8 decoded: {ctc(uint8[0])}")
    else:
        print(f"vocab.json not found at {vocab_path}")
else:
    print("No output files found")
    if os.path.exists(fp32_dir):
        print(f"FP32 dir: {os.listdir(fp32_dir)}")
    if os.path.exists(uint8_dir):
        print(f"uint8 dir: {os.listdir(uint8_dir)}")
PYTHON_COMPARE

echo ""
echo "=== Pipeline complete ==="
echo "Quantize file: ${WKSP}/${MODEL_NAME}_uint8.quantize"
echo ""
echo "To export NB (run in Docker):"
echo "pegasus export ovxlib \\"
echo "  --model ${MODEL_NAME}.json \\"
echo "  --model-data ${MODEL_NAME}.data \\"
echo "  --model-quantize ${MODEL_NAME}_uint8.quantize \\"
echo "  --dtype quantized \\"
echo "  --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016 \\"
echo "  --viv-sdk \$VIV_SDK/vsimulator --target-ide-project linux64 --batch-size 1 \\"
echo "  --output-path ${WKSP}/${MODEL_NAME}_uint8_nbg_unify/"

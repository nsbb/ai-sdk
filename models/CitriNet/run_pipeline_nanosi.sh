#!/usr/bin/env bash
###############################################################################
# Citrinet VIP9000NANOSI_PLUS NPU 컴파일 파이프라인
# Target: VIP9000NANOSI_PLUS_PID0X10000016
###############################################################################
set -euo pipefail

# ==============================
# 설정
# ==============================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIVANTE_SDK="/home/nsbb/VeriSilicon/VivanteIDE5.7.2/cmdtools/vsimulator"
ACUITY_BIN="${SCRIPT_DIR}/../../../bin/pegasus"  # 경로 조정 필요

ONNX_MODEL="${SCRIPT_DIR}/citrinet_npu.onnx"
MODEL_NAME="citrinet_npu"
INPUT_NAME="audio_signal"
OUTPUT_NAME="logits"
INPUT_SHAPE="1,80,1,300"
QTYPE="uint8"

# 중요: 타겟 명시
NPU_TARGET="VIP9000NANOSI_PLUS_PID0X10000016"

CALIB_LIST="${SCRIPT_DIR}/calib_dataset.txt"
OUT_DIR="${SCRIPT_DIR}/wksp"
EXPORT_DIR="${OUT_DIR}/citrinet_npu_${QTYPE}"

# 환경 변수 설정
export VIVANTE_SDK_DIR="${VIVANTE_SDK}"
export LD_LIBRARY_PATH="${VIVANTE_SDK}/lib:${VIVANTE_SDK}/lib/x64_linux:${VIVANTE_SDK}/lib/x64_linux/vsim:${LD_LIBRARY_PATH:-}"

echo "======================================================================"
echo " Citrinet VIP9000NANOSI_PLUS NPU 컴파일"
echo "======================================================================"
echo " Target      : ${NPU_TARGET}"
echo " ONNX Model  : ${ONNX_MODEL}"
echo " Quant Type  : ${QTYPE}"
echo " Output Dir  : ${OUT_DIR}"
echo "======================================================================"

# 기존 출력 삭제
echo ""
echo "[Clean] 기존 출력 파일 삭제..."
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# ==============================
# Step 1: Import
# ==============================
echo ""
echo "[Step 1/4] Pegasus Import ONNX"
echo "----------------------------------------------------------------------"

${ACUITY_BIN} import onnx \
    --model "${ONNX_MODEL}" \
    --inputs "${INPUT_NAME}" \
    --outputs "${OUTPUT_NAME}" \
    --input-size-list "${INPUT_SHAPE}" \
    --size-with-batch "True" \
    --output-model "${OUT_DIR}/${MODEL_NAME}.json" \
    --output-data "${OUT_DIR}/${MODEL_NAME}.data"

echo "✅ Import 완료"

# ==============================
# Step 2: InputMeta
# ==============================
echo ""
echo "[Step 2/4] InputMeta 생성"
echo "----------------------------------------------------------------------"

cat > "${OUT_DIR}/${MODEL_NAME}_inputmeta.yml" <<EOF
InputMeta:
  enable_preprocess: false
  enable_crop: false
  inputs:
    - name: ${INPUT_NAME}
      dtype: float32
      shape: [1, 80, 1, 300]
      layout: NCHW
      category: undefined
      path: $(realpath ${CALIB_LIST})
EOF

echo "✅ InputMeta 생성 완료"

# ==============================
# Step 3: Quantize - 타겟 명시!
# ==============================
echo ""
echo "[Step 3/4] Pegasus Quantize (${QTYPE}) - Target: ${NPU_TARGET}"
echo "----------------------------------------------------------------------"

${ACUITY_BIN} quantize \
    --model "${OUT_DIR}/${MODEL_NAME}.json" \
    --model-data "${OUT_DIR}/${MODEL_NAME}.data" \
    --with-input-meta "${OUT_DIR}/${MODEL_NAME}_inputmeta.yml" \
    --device CPU \
    --quantizer asymmetric_affine \
    --qtype "${QTYPE}" \
    --rebuild-all \
    --algorithm normal \
    --optimize "${NPU_TARGET}" \
    --model-quantize "${OUT_DIR}/${MODEL_NAME}_${QTYPE}.quantize"

echo "✅ Quantize 완료"

# ==============================
# Step 4: Export - 동일한 타겟!
# ==============================
echo ""
echo "[Step 4/4] Pegasus Export OVXLib - Target: ${NPU_TARGET}"
echo "----------------------------------------------------------------------"

${ACUITY_BIN} export ovxlib \
    --model "${OUT_DIR}/${MODEL_NAME}.json" \
    --model-data "${OUT_DIR}/${MODEL_NAME}.data" \
    --model-quantize "${OUT_DIR}/${MODEL_NAME}_${QTYPE}.quantize" \
    --with-input-meta "${OUT_DIR}/${MODEL_NAME}_inputmeta.yml" \
    --dtype quantized \
    --viv-sdk "${VIVANTE_SDK}" \
    --pack-nbg-unify \
    --optimize "${NPU_TARGET}" \
    --target-ide-project linux64 \
    --output-path "${EXPORT_DIR}/" \
    --batch-size 1

echo ""
echo "======================================================================"
echo "✅ 컴파일 완료!"
echo "======================================================================"
echo ""
echo "NBG 파일 위치:"
find "${OUT_DIR}" -name "*.nb" -o -name "network_binary.nb" 2>/dev/null
echo ""
echo "타겟 확인:"
echo "  설정한 타겟: ${NPU_TARGET} (0x10000016)"
echo "======================================================================"


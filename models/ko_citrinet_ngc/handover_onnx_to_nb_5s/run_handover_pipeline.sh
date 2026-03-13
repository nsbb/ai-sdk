#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -----------------------------------------------------------------------------
# User-tunable variables
# -----------------------------------------------------------------------------
NEMO_ENV="${NEMO_ENV:-nemo_py310}"
ACUITY_BIN="${ACUITY_BIN:-/nas02/geonhui83/T527_toolkit/acuity-toolkit-binary-6.12.0/bin}"
PEGASUS="${ACUITY_BIN}/pegasus"

MODEL_NEMO="${MODEL_NEMO:-/nas02/geonhui83/stt/citrinet_korean/Citrinet-1024-gamma-0.25_spe-2048_ko-KR_Riva-ASR-SET-1.0.nemo}"
ONNX_PATH="${ONNX_PATH:-}"
ONNX_URL="${ONNX_URL:-}"
ONNX_NAME="${ONNX_NAME:-citrinet_handover.onnx}"

CALIB_LIST="${CALIB_LIST:-}"
CALIB_MANIFEST="${CALIB_MANIFEST:-${ROOT_DIR}/ko_citrinet_ngc/data/calib_manifest.tsv}"
TEST_MANIFEST="${TEST_MANIFEST:-${ROOT_DIR}/ko_citrinet_ngc/data/test_manifest.tsv}"

TIME_FRAMES="${TIME_FRAMES:-500}"
QTYPE="${QTYPE:-int8}"
QUANTIZER="${QUANTIZER:-asymmetric_affine}"
ALGORITHM="${ALGORITHM:-moving_average}"
MA_WEIGHT="${MA_WEIGHT:-0.004}"
USE_MLE="${USE_MLE:-false}"
INPUT_REVERSE_CHANNEL="${INPUT_REVERSE_CHANNEL:-false}"
OPTIMIZE="${OPTIMIZE:-VIP9000NANOSI_PLUS_PID0X10000016}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts}"
BUNDLE_INT8="${BUNDLE_INT8:-${SCRIPT_DIR}/bundle_int8}"
BUNDLE_FP32="${BUNDLE_FP32:-${SCRIPT_DIR}/bundle_fp32}"

VIVANTE_BASE="${VIVANTE_BASE:-/home1/Gunhee_Lee/VeriSilicon/VivanteIDE5.7.2}"
VIV_SDK="${VIVANTE_BASE}/cmdtools/vsimulator"
COMMON_LIB="${VIVANTE_BASE}/cmdtools/common/lib"
SIM_LIB="${VIV_SDK}/lib"
SIM_X64_LIB="${VIV_SDK}/lib/x64_linux"
SIM_X64_VSIM_LIB="${VIV_SDK}/lib/x64_linux/vsim"

# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
if [[ ! -x "${PEGASUS}" ]]; then
  echo "ERROR: pegasus not found: ${PEGASUS}" >&2
  exit 1
fi
if [[ -z "${ONNX_PATH}" && -z "${ONNX_URL}" && ! -f "${MODEL_NEMO}" ]]; then
  echo "ERROR: set ONNX_PATH / ONNX_URL or valid MODEL_NEMO=${MODEL_NEMO}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${ARTIFACT_DIR}" "${BUNDLE_INT8}" "${BUNDLE_FP32}"

source /home1/Gunhee_Lee/anaconda3/etc/profile.d/conda.sh
conda activate "${NEMO_ENV}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER:-user}}"
mkdir -p "${NUMBA_CACHE_DIR}"

ONNX_OUT="${WORK_DIR}/${ONNX_NAME}"

# -----------------------------------------------------------------------------
# 1) ONNX 준비 (로컬 파일 / URL 다운로드 / .nemo export)
# -----------------------------------------------------------------------------
echo "[1/9] Prepare ONNX"
if [[ -n "${ONNX_PATH}" ]]; then
  if [[ ! -f "${ONNX_PATH}" ]]; then
    echo "ERROR: ONNX_PATH not found: ${ONNX_PATH}" >&2
    exit 1
  fi
  cp -f "${ONNX_PATH}" "${ONNX_OUT}"
  echo " - ONNX from local path"
elif [[ -n "${ONNX_URL}" ]]; then
  echo " - ONNX download from URL"
  curl -fL "${ONNX_URL}" -o "${ONNX_OUT}"
else
  echo " - ONNX export from .nemo"
  python "${ROOT_DIR}/ko_citrinet_ngc/export_onnx_ko_v2.py" \
    --model-file "${MODEL_NEMO}" \
    --output-dir "${WORK_DIR}" \
    --onnx-name "${ONNX_NAME}" \
    --time-frames "${TIME_FRAMES}"
fi

if [[ ! -f "${ONNX_OUT}" ]]; then
  echo "ERROR: onnx not prepared: ${ONNX_OUT}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# 2) Calibration/Test npy list 준비
# -----------------------------------------------------------------------------
echo "[2/9] Prepare calibration/test datasets"
if [[ -n "${CALIB_LIST}" ]]; then
  if [[ ! -f "${CALIB_LIST}" ]]; then
    echo "ERROR: CALIB_LIST not found: ${CALIB_LIST}" >&2
    exit 1
  fi
  CALIB_DATASET="${CALIB_LIST}"
else
  python "${ROOT_DIR}/ko_citrinet_ngc/make_npy_dataset.py" \
    --manifest-tsv "${CALIB_MANIFEST}" \
    --out-dir "${WORK_DIR}/calib_npy" \
    --out-list "${WORK_DIR}/calib_dataset.txt" \
    --model-file "${MODEL_NEMO}" \
    --time-frames "${TIME_FRAMES}"
  CALIB_DATASET="${WORK_DIR}/calib_dataset.txt"
fi

python "${ROOT_DIR}/ko_citrinet_ngc/make_npy_dataset.py" \
  --manifest-tsv "${TEST_MANIFEST}" \
  --out-dir "${WORK_DIR}/test_npy" \
  --out-list "${WORK_DIR}/test_dataset.txt" \
  --model-file "${MODEL_NEMO}" \
  --time-frames "${TIME_FRAMES}"

TEST_DATASET="${WORK_DIR}/test_dataset.txt"
TEST_META="${WORK_DIR}/test_dataset.meta.tsv"
CALIB_ITERS="$(awk 'NF{c++} END{print c+0}' "${CALIB_DATASET}")"
if [[ "${CALIB_ITERS}" -lt 1 ]]; then
  echo "ERROR: empty calibration list: ${CALIB_DATASET}" >&2
  exit 1
fi

FIRST_TEST_NPY="$(head -n 1 "${TEST_DATASET}")"
if [[ -z "${FIRST_TEST_NPY}" || ! -f "${FIRST_TEST_NPY}" ]]; then
  echo "ERROR: no test npy generated" >&2
  exit 1
fi

FIRST_TEST_WAV="$(awk -F '\t' 'NR==2{print $2}' "${TEST_META}")"
FIRST_TEST_TEXT="$(awk -F '\t' 'NR==2{print $3}' "${TEST_META}")"

# -----------------------------------------------------------------------------
# 3) Import ONNX
# -----------------------------------------------------------------------------
echo "[3/9] Pegasus import onnx"
"${PEGASUS}" import onnx \
  --model "${ONNX_OUT}" \
  --inputs "audio_signal" \
  --outputs "logits" \
  --input-size-list "1,80,1,${TIME_FRAMES}" \
  --size-with-batch "True" \
  --output-model "${WORK_DIR}/citrinet_handover.json" \
  --output-data "${WORK_DIR}/citrinet_handover.data"

# -----------------------------------------------------------------------------
# 4) Inputmeta 생성/패치
# -----------------------------------------------------------------------------
echo "[4/9] Generate inputmeta"
"${PEGASUS}" generate inputmeta \
  --model "${WORK_DIR}/citrinet_handover.json" \
  --input-meta-output "${WORK_DIR}/citrinet_handover_inputmeta.yml"

python - <<PY
from pathlib import Path
p = Path("${WORK_DIR}/citrinet_handover_inputmeta.yml")
lines = p.read_text(encoding="utf-8").splitlines()
out = []
for line in lines:
    s = line.strip()
    indent = line[: len(line) - len(line.lstrip())]
    if s.startswith("path: "):
        out.append(f"{indent}path: ${CALIB_DATASET}")
    elif s.startswith("- path: "):
        out.append(f"{indent}- path: ${CALIB_DATASET}")
    elif s.startswith("category: "):
        out.append(f"{indent}category: undefined")
    elif s.startswith("reverse_channel:"):
        out.append(f"{indent}reverse_channel: ${INPUT_REVERSE_CHANNEL}")
    else:
        out.append(line)
p.write_text("\\n".join(out) + "\\n", encoding="utf-8")
PY

awk '
BEGIN { replacing=0 }
/^[[:space:]]*mean:[[:space:]]*$/ {
    print
    for (i=1; i<=80; i++) print "        - 0"
    replacing=1
    next
}
replacing==1 {
    if ($0 ~ /^[[:space:]]*-[[:space:]]*/) next
    replacing=0
}
{ print }
' "${WORK_DIR}/citrinet_handover_inputmeta.yml" > "${WORK_DIR}/citrinet_handover_inputmeta.tmp.yml"
mv "${WORK_DIR}/citrinet_handover_inputmeta.tmp.yml" "${WORK_DIR}/citrinet_handover_inputmeta.yml"

# -----------------------------------------------------------------------------
# 5) INT8 Quantize
# -----------------------------------------------------------------------------
echo "[5/9] Quantize (${QTYPE}, ${ALGORITHM}, ${QUANTIZER})"
QFILE="${WORK_DIR}/citrinet_handover_${QTYPE}.quantize"
QCMD=(
  "${PEGASUS}" quantize
  --model "${WORK_DIR}/citrinet_handover.json"
  --model-data "${WORK_DIR}/citrinet_handover.data"
  --with-input-meta "${WORK_DIR}/citrinet_handover_inputmeta.yml"
  --iterations "${CALIB_ITERS}"
  --device CPU
  --quantizer "${QUANTIZER}"
  --qtype "${QTYPE}"
  --rebuild-all
  --algorithm "${ALGORITHM}"
  --model-quantize "${QFILE}"
)
if [[ "${ALGORITHM}" == "moving_average" && -n "${MA_WEIGHT}" ]]; then
  QCMD+=(--moving-average-weight "${MA_WEIGHT}")
fi
if [[ "${USE_MLE}" == "true" ]]; then
  QCMD+=(--MLE)
fi
"${QCMD[@]}"

# -----------------------------------------------------------------------------
# 6) INT8/FP32 NB Export
# -----------------------------------------------------------------------------
echo "[6/9] Export INT8/FP32 NB"
export VIVANTE_VIP_HOME="${VIVANTE_BASE}/"
export VIVANTE_SDK_DIR="${VIV_SDK}"
export LD_LIBRARY_PATH="${SIM_LIB}:${COMMON_LIB}:${SIM_X64_LIB}:${SIM_X64_VSIM_LIB}:${LD_LIBRARY_PATH:-}"
export REAL_GCC="/usr/bin/gcc"
export PATCHELF_BIN="/home1/Gunhee_Lee/anaconda3/bin/patchelf"
export FORCE_RPATH="${SIM_LIB}:${COMMON_LIB}:${SIM_X64_LIB}:${SIM_X64_VSIM_LIB}"
export EXTRALFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,${SIM_LIB} -Wl,-rpath,${COMMON_LIB} -Wl,-rpath,${SIM_X64_LIB} -Wl,-rpath,${SIM_X64_VSIM_LIB}"

"${PEGASUS}" export ovxlib \
  --model "${WORK_DIR}/citrinet_handover.json" \
  --model-data "${WORK_DIR}/citrinet_handover.data" \
  --model-quantize "${QFILE}" \
  --with-input-meta "${WORK_DIR}/citrinet_handover_inputmeta.yml" \
  --dtype quantized \
  --viv-sdk "${VIV_SDK}" \
  --pack-nbg-unify \
  --optimize "${OPTIMIZE}" \
  --target-ide-project linux64 \
  --output-path "${ARTIFACT_DIR}/int8/output/" \
  --batch-size 1

"${PEGASUS}" export ovxlib \
  --model "${WORK_DIR}/citrinet_handover.json" \
  --model-data "${WORK_DIR}/citrinet_handover.data" \
  --with-input-meta "${WORK_DIR}/citrinet_handover_inputmeta.yml" \
  --dtype float32 \
  --viv-sdk "${VIV_SDK}" \
  --pack-nbg-unify \
  --optimize "${OPTIMIZE}" \
  --target-ide-project linux64 \
  --output-path "${ARTIFACT_DIR}/fp32/output/" \
  --batch-size 1

# -----------------------------------------------------------------------------
# 7) Bundle 생성
# -----------------------------------------------------------------------------
echo "[7/9] Build handover bundles"
cp -f "${ARTIFACT_DIR}/int8/output_nbg_unify/network_binary.nb" "${BUNDLE_INT8}/network_binary.nb"
cp -f "${ARTIFACT_DIR}/int8/output_nbg_unify/nbg_meta.json" "${BUNDLE_INT8}/nbg_meta.json"

cp -f "${ARTIFACT_DIR}/fp32/output_nbg_unify/network_binary.nb" "${BUNDLE_FP32}/network_binary.nb"
cp -f "${ARTIFACT_DIR}/fp32/output_nbg_unify/nbg_meta.json" "${BUNDLE_FP32}/nbg_meta.json"

cat > "${BUNDLE_INT8}/sample.txt" <<'EOF'
[network]
/data/local/tmp/citrinet_ko_handover_int8/network_binary.nb
[input]
/data/local/tmp/citrinet_ko_handover_int8/input_0.dat
[output]
/data/local/tmp/citrinet_ko_handover_int8/output_0.dat
EOF

cat > "${BUNDLE_FP32}/sample.txt" <<'EOF'
[network]
/data/local/tmp/citrinet_ko_handover_fp32/network_binary.nb
[input]
/data/local/tmp/citrinet_ko_handover_fp32/input_0.dat
[output]
/data/local/tmp/citrinet_ko_handover_fp32/output_0.dat
EOF

# -----------------------------------------------------------------------------
# 8) 5초 입력 생성 (실제 wav 5초로 패딩/트림 후 feature 생성)
# -----------------------------------------------------------------------------
echo "[8/9] Create real 5-second input"
SOURCE_5S_WAV="${SOURCE_5S_WAV:-${FIRST_TEST_WAV}}"
if [[ -z "${SOURCE_5S_WAV}" || ! -f "${SOURCE_5S_WAV}" ]]; then
  echo "ERROR: SOURCE_5S_WAV not found: ${SOURCE_5S_WAV}" >&2
  exit 1
fi

python "${SCRIPT_DIR}/make_5s_input.py" \
  --wav "${SOURCE_5S_WAV}" \
  --model-file "${MODEL_NEMO}" \
  --meta "${BUNDLE_INT8}/nbg_meta.json" \
  --out-wav "${BUNDLE_INT8}/test_input_5s.wav" \
  --out-float-npy "${BUNDLE_INT8}/input_5s_float.npy" \
  --out-dat "${BUNDLE_INT8}/input_5s.dat" \
  --out-fp32-dat "${BUNDLE_FP32}/input_5s.dat" \
  --duration-sec 5.0 \
  --time-frames "${TIME_FRAMES}"

cp -f "${BUNDLE_INT8}/test_input_5s.wav" "${BUNDLE_FP32}/test_input_5s.wav"
printf "%s\n" "${FIRST_TEST_TEXT}" > "${BUNDLE_INT8}/test_gt.txt"
printf "%s\n" "${FIRST_TEST_TEXT}" > "${BUNDLE_FP32}/test_gt.txt"

# vpm default name aliases
cp -f "${BUNDLE_INT8}/input_5s.dat" "${BUNDLE_INT8}/input_0.dat"
cp -f "${BUNDLE_FP32}/input_5s.dat" "${BUNDLE_FP32}/input_0.dat"
cp -f "${BUNDLE_INT8}/input_5s_float.npy" "${BUNDLE_FP32}/input_5s_float.npy"

# -----------------------------------------------------------------------------
# 9) Done
# -----------------------------------------------------------------------------
echo "[9/9] Done"
echo "======================================================================"
echo "ONNX          : ${ONNX_OUT}"
echo "INT8 NB       : ${BUNDLE_INT8}/network_binary.nb"
echo "FP32 NB       : ${BUNDLE_FP32}/network_binary.nb"
echo "INT8 input    : ${BUNDLE_INT8}/input_5s.dat"
echo "FP32 input    : ${BUNDLE_FP32}/input_5s.dat"
echo "WAV (5 sec)   : ${BUNDLE_INT8}/test_input_5s.wav"
echo "SAMPLE WAV src: ${SOURCE_5S_WAV}"
echo "======================================================================"

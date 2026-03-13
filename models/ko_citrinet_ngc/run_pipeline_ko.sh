#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACUITY_BIN="${ACUITY_BIN:-/nas02/geonhui83/T527_toolkit/acuity-toolkit-binary-6.12.0/bin}"
PEGASUS="${ACUITY_BIN}/pegasus"
NEMO_ENV="${NEMO_ENV:-nemo_py310}"

MODEL_NEMO="${MODEL_NEMO:-}"
MODEL_NAME="${MODEL_NAME:-}"
TRAIN_CSV="${TRAIN_CSV:-/nas04/nlp_sk/STT/data/train/base/train_base_4356hr.csv}"
WAV_COL="${WAV_COL:-raw_data}"
TEXT_COL="${TEXT_COL:-transcript}"

CALIB_COUNT="${CALIB_COUNT:-120}"
TEST_COUNT="${TEST_COUNT:-20}"
SEED="${SEED:-42}"
TIME_FRAMES="${TIME_FRAMES:-300}"
MAX_ROWS="${MAX_ROWS:-0}"

QTYPE="${QTYPE:-int8}"
ALGORITHM="${ALGORITHM:-moving_average}"
QUANTIZER="${QUANTIZER:-asymmetric_affine}"
MA_WEIGHT="${MA_WEIGHT:-0.004}"
USE_MLE="${USE_MLE:-false}"
INPUT_REVERSE_CHANNEL="${INPUT_REVERSE_CHANNEL:-false}"
OPTIMIZE="${OPTIMIZE:-VIP9000NANOSI_PLUS_PID0X10000016}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts}"
BUNDLE_INT8="${BUNDLE_INT8:-${SCRIPT_DIR}/bundle_int8}"
BUNDLE_FP32="${BUNDLE_FP32:-${SCRIPT_DIR}/bundle_fp32}"

VIVANTE_BASE="${VIVANTE_BASE:-/home1/Gunhee_Lee/VeriSilicon/VivanteIDE5.7.2}"
VIV_SDK="${VIVANTE_BASE}/cmdtools/vsimulator"
COMMON_LIB="${VIVANTE_BASE}/cmdtools/common/lib"
SIM_LIB="${VIV_SDK}/lib"
SIM_X64_LIB="${VIV_SDK}/lib/x64_linux"
SIM_X64_VSIM_LIB="${VIV_SDK}/lib/x64_linux/vsim"

if [[ ! -x "${PEGASUS}" ]]; then
  echo "ERROR: pegasus not found: ${PEGASUS}" >&2
  exit 1
fi
if [[ -z "${MODEL_NEMO}" && -z "${MODEL_NAME}" ]]; then
  echo "ERROR: set MODEL_NEMO=<path/to/model.nemo> or MODEL_NAME=<nemo_pretrained_name>" >&2
  exit 1
fi
if [[ -n "${MODEL_NEMO}" && ! -f "${MODEL_NEMO}" ]]; then
  echo "ERROR: model file not found: ${MODEL_NEMO}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${DATA_DIR}" "${ARTIFACT_DIR}" "${BUNDLE_INT8}" "${BUNDLE_FP32}"

source /home1/Gunhee_Lee/anaconda3/etc/profile.d/conda.sh
conda activate "${NEMO_ENV}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER:-user}}"
mkdir -p "${NUMBA_CACHE_DIR}"

MODEL_ARGS=()
if [[ -n "${MODEL_NEMO}" ]]; then
  MODEL_ARGS+=(--model-file "${MODEL_NEMO}")
else
  MODEL_ARGS+=(--model-name "${MODEL_NAME}")
fi

echo "======================================================================"
echo " Korean Citrinet FP32/INT8 NB pipeline"
echo "======================================================================"
echo " TRAIN_CSV     : ${TRAIN_CSV}"
echo " CALIB/TEST    : ${CALIB_COUNT}/${TEST_COUNT}"
echo " QTYPE         : ${QTYPE}"
echo " ALGORITHM     : ${ALGORITHM}"
echo " QUANTIZER     : ${QUANTIZER}"
echo " MA_WEIGHT     : ${MA_WEIGHT}"
echo " WORK_DIR      : ${WORK_DIR}"
echo " DATA_DIR      : ${DATA_DIR}"
echo " ARTIFACT_DIR  : ${ARTIFACT_DIR}"
echo "======================================================================"

echo "[1/9] Copy wav subset from CSV"
python "${SCRIPT_DIR}/prepare_ko_wav_from_csv.py" \
  --csv-path "${TRAIN_CSV}" \
  --wav-col "${WAV_COL}" \
  --text-col "${TEXT_COL}" \
  --calib-count "${CALIB_COUNT}" \
  --test-count "${TEST_COUNT}" \
  --seed "${SEED}" \
  --max-rows "${MAX_ROWS}" \
  --out-dir "${DATA_DIR}"

echo "[2/9] Export ONNX from Korean Citrinet model"
python "${SCRIPT_DIR}/export_onnx_ko.py" \
  "${MODEL_ARGS[@]}" \
  --output-dir "${WORK_DIR}" \
  --time-frames "${TIME_FRAMES}"

echo "[3/9] Build calibration/test npy datasets"
python "${SCRIPT_DIR}/make_npy_dataset.py" \
  --manifest-tsv "${DATA_DIR}/calib_manifest.tsv" \
  --out-dir "${WORK_DIR}/calib_npy" \
  --out-list "${WORK_DIR}/calib_dataset.txt" \
  "${MODEL_ARGS[@]}" \
  --time-frames "${TIME_FRAMES}"

python "${SCRIPT_DIR}/make_npy_dataset.py" \
  --manifest-tsv "${DATA_DIR}/test_manifest.tsv" \
  --out-dir "${WORK_DIR}/test_npy" \
  --out-list "${WORK_DIR}/test_dataset.txt" \
  "${MODEL_ARGS[@]}" \
  --time-frames "${TIME_FRAMES}"

CALIB_LIST="${WORK_DIR}/calib_dataset.txt"
TEST_LIST="${WORK_DIR}/test_dataset.txt"
CALIB_ITERS="$(awk 'NF{c++} END{print c+0}' "${CALIB_LIST}")"
if [[ "${CALIB_ITERS}" -lt 1 ]]; then
  echo "ERROR: calibration list is empty: ${CALIB_LIST}" >&2
  exit 1
fi

echo "[4/9] Pegasus import onnx"
"${PEGASUS}" import onnx \
  --model "${WORK_DIR}/citrinet_npu.onnx" \
  --inputs "audio_signal" \
  --outputs "logits" \
  --input-size-list "1,80,1,${TIME_FRAMES}" \
  --size-with-batch "True" \
  --output-model "${WORK_DIR}/citrinet_npu.json" \
  --output-data "${WORK_DIR}/citrinet_npu.data"

echo "[5/9] Generate and patch inputmeta"
"${PEGASUS}" generate inputmeta \
  --model "${WORK_DIR}/citrinet_npu.json" \
  --input-meta-output "${WORK_DIR}/citrinet_npu_inputmeta.yml"

python - <<PY
from pathlib import Path
p = Path("${WORK_DIR}/citrinet_npu_inputmeta.yml")
lines = p.read_text().splitlines()
out = []
for line in lines:
    s = line.strip()
    indent = line[: len(line) - len(line.lstrip())]
    if s.startswith("path: "):
        out.append(f"{indent}path: ${CALIB_LIST}")
    elif s.startswith("- path: "):
        out.append(f"{indent}- path: ${CALIB_LIST}")
    elif s.startswith("category: "):
        out.append(f"{indent}category: undefined")
    elif s.startswith("reverse_channel:"):
        out.append(f"{indent}reverse_channel: ${INPUT_REVERSE_CHANNEL}")
    else:
        out.append(line)
p.write_text("\\n".join(out) + "\\n")
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
' "${WORK_DIR}/citrinet_npu_inputmeta.yml" > "${WORK_DIR}/citrinet_npu_inputmeta.tmp.yml"
mv "${WORK_DIR}/citrinet_npu_inputmeta.tmp.yml" "${WORK_DIR}/citrinet_npu_inputmeta.yml"

echo "[6/9] Quantize INT8"
QFILE="${WORK_DIR}/citrinet_npu_${QTYPE}.quantize"
QCMD=(
  "${PEGASUS}" quantize
  --model "${WORK_DIR}/citrinet_npu.json"
  --model-data "${WORK_DIR}/citrinet_npu.data"
  --with-input-meta "${WORK_DIR}/citrinet_npu_inputmeta.yml"
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

export VIVANTE_VIP_HOME="${VIVANTE_BASE}/"
export VIVANTE_SDK_DIR="${VIV_SDK}"
export LD_LIBRARY_PATH="${SIM_LIB}:${COMMON_LIB}:${SIM_X64_LIB}:${SIM_X64_VSIM_LIB}:${LD_LIBRARY_PATH:-}"
export REAL_GCC="/usr/bin/gcc"
export PATCHELF_BIN="/home1/Gunhee_Lee/anaconda3/bin/patchelf"
export FORCE_RPATH="${SIM_LIB}:${COMMON_LIB}:${SIM_X64_LIB}:${SIM_X64_VSIM_LIB}"
export EXTRALFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,${SIM_LIB} -Wl,-rpath,${COMMON_LIB} -Wl,-rpath,${SIM_X64_LIB} -Wl,-rpath,${SIM_X64_VSIM_LIB}"

echo "[7/9] Export INT8/FP32 NBG"
"${PEGASUS}" export ovxlib \
  --model "${WORK_DIR}/citrinet_npu.json" \
  --model-data "${WORK_DIR}/citrinet_npu.data" \
  --model-quantize "${QFILE}" \
  --with-input-meta "${WORK_DIR}/citrinet_npu_inputmeta.yml" \
  --dtype quantized \
  --viv-sdk "${VIV_SDK}" \
  --pack-nbg-unify \
  --optimize "${OPTIMIZE}" \
  --target-ide-project linux64 \
  --output-path "${ARTIFACT_DIR}/int8/output/" \
  --batch-size 1

"${PEGASUS}" export ovxlib \
  --model "${WORK_DIR}/citrinet_npu.json" \
  --model-data "${WORK_DIR}/citrinet_npu.data" \
  --with-input-meta "${WORK_DIR}/citrinet_npu_inputmeta.yml" \
  --dtype float32 \
  --viv-sdk "${VIV_SDK}" \
  --pack-nbg-unify \
  --optimize "${OPTIMIZE}" \
  --target-ide-project linux64 \
  --output-path "${ARTIFACT_DIR}/fp32/output/" \
  --batch-size 1

FIRST_TEST_NPY="$(head -n 1 "${TEST_LIST}")"
if [[ ! -f "${FIRST_TEST_NPY}" ]]; then
  echo "ERROR: first test npy missing: ${FIRST_TEST_NPY}" >&2
  exit 1
fi
cp -f "${FIRST_TEST_NPY}" "${ARTIFACT_DIR}/test_input_0_float.npy"

echo "[8/9] Build input_0.dat for INT8/FP32"
python "${ROOT_DIR}/android13_t527_vpm_run/requantize_input_from_npy.py" \
  --float-npy "${ARTIFACT_DIR}/test_input_0_float.npy" \
  --meta "${ARTIFACT_DIR}/int8/output_nbg_unify/nbg_meta.json" \
  --out-dat "${ARTIFACT_DIR}/int8/output_nbg_unify/input_0.dat"

python - <<PY
import numpy as np
x = np.load("${ARTIFACT_DIR}/test_input_0_float.npy").astype(np.float32, copy=False)
x.tofile("${ARTIFACT_DIR}/fp32/output_nbg_unify/input_0.dat")
print("[OK] wrote fp32 input dat")
PY

FIRST_TEST_WAV="$(awk -F'\t' '$1==\"wav_path\"{print $2}' "${DATA_DIR}/test_first.txt")"
FIRST_TEST_TEXT="$(awk -F'\t' '$1==\"text\"{print $2}' "${DATA_DIR}/test_first.txt")"

echo "[9/9] Build vpm bundles"
mkdir -p "${BUNDLE_INT8}" "${BUNDLE_FP32}"

cat > "${BUNDLE_INT8}/sample.txt" <<'EOF'
[network]
/data/local/tmp/citrinet_ko_int8/network_binary.nb
[input]
/data/local/tmp/citrinet_ko_int8/input_0.dat
[output]
/data/local/tmp/citrinet_ko_int8/output_0.dat
EOF

cat > "${BUNDLE_FP32}/sample.txt" <<'EOF'
[network]
/data/local/tmp/citrinet_ko_fp32/network_binary.nb
[input]
/data/local/tmp/citrinet_ko_fp32/input_0.dat
[output]
/data/local/tmp/citrinet_ko_fp32/output_0.dat
EOF

cp -f "${ARTIFACT_DIR}/int8/output_nbg_unify/network_binary.nb" "${BUNDLE_INT8}/network_binary.nb"
cp -f "${ARTIFACT_DIR}/int8/output_nbg_unify/input_0.dat" "${BUNDLE_INT8}/input_0.dat"
cp -f "${ARTIFACT_DIR}/int8/output_nbg_unify/nbg_meta.json" "${BUNDLE_INT8}/nbg_meta.json"

cp -f "${ARTIFACT_DIR}/fp32/output_nbg_unify/network_binary.nb" "${BUNDLE_FP32}/network_binary.nb"
cp -f "${ARTIFACT_DIR}/fp32/output_nbg_unify/input_0.dat" "${BUNDLE_FP32}/input_0.dat"
cp -f "${ARTIFACT_DIR}/fp32/output_nbg_unify/nbg_meta.json" "${BUNDLE_FP32}/nbg_meta.json"

if [[ -n "${FIRST_TEST_WAV}" && -f "${FIRST_TEST_WAV}" ]]; then
  cp -f "${FIRST_TEST_WAV}" "${BUNDLE_INT8}/test_input.wav"
  cp -f "${FIRST_TEST_WAV}" "${BUNDLE_FP32}/test_input.wav"
fi
printf "%s\n" "${FIRST_TEST_TEXT}" > "${BUNDLE_INT8}/test_gt.txt"
printf "%s\n" "${FIRST_TEST_TEXT}" > "${BUNDLE_FP32}/test_gt.txt"

echo "======================================================================"
echo "[DONE] Korean Citrinet pipeline finished"
echo " INT8 NB : ${ARTIFACT_DIR}/int8/output_nbg_unify/network_binary.nb"
echo " FP32 NB : ${ARTIFACT_DIR}/fp32/output_nbg_unify/network_binary.nb"
echo " Bundle INT8: ${BUNDLE_INT8}"
echo " Bundle FP32: ${BUNDLE_FP32}"
echo ""
echo "ADB example (INT8):"
echo "  adb shell 'mkdir -p /data/local/tmp/citrinet_ko_int8'"
echo "  adb push ${BUNDLE_INT8}/network_binary.nb /data/local/tmp/citrinet_ko_int8/"
echo "  adb push ${BUNDLE_INT8}/input_0.dat /data/local/tmp/citrinet_ko_int8/"
echo "  adb push ${BUNDLE_INT8}/sample.txt /data/local/tmp/citrinet_ko_int8/"
echo "======================================================================"

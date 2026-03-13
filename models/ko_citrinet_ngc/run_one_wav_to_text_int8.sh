#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /abs/path/test.wav [GT_TEXT]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WAV_PATH="$(realpath "$1")"
GT_TEXT="${2:-}"

MODEL_FILE="${MODEL_FILE:-/nas02/geonhui83/stt/citrinet_korean/Citrinet-1024-gamma-0.25_spe-2048_ko-KR_Riva-ASR-SET-1.0.nemo}"
NEMO_ENV="${NEMO_ENV:-nemo_py310}"
ADB_BIN="${ADB_BIN:-adb}"
VPM_BIN_ON_DEVICE="${VPM_BIN_ON_DEVICE:-/data/local/tmp/vpm_run_test/vpm_run}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/citrinet_ko_int8}"

NB_PATH="${NB_PATH:-${SCRIPT_DIR}/artifacts/int8/output_nbg_unify/network_binary.nb}"
META_PATH="${META_PATH:-${SCRIPT_DIR}/artifacts/int8/output_nbg_unify/nbg_meta.json}"

RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/one_wav_runtime}"
mkdir -p "${RUN_DIR}/npy"

if [[ ! -f "${WAV_PATH}" ]]; then
  echo "ERROR: wav not found: ${WAV_PATH}"
  exit 1
fi
if [[ ! -f "${NB_PATH}" ]]; then
  echo "ERROR: nb not found: ${NB_PATH}"
  exit 1
fi
if [[ ! -f "${META_PATH}" ]]; then
  echo "ERROR: meta not found: ${META_PATH}"
  exit 1
fi
if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: model file not found: ${MODEL_FILE}"
  exit 1
fi

source /home1/Gunhee_Lee/anaconda3/etc/profile.d/conda.sh
conda activate "${NEMO_ENV}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER:-user}}"
mkdir -p "${NUMBA_CACHE_DIR}"

echo "[1/6] wav -> npy feature"
MANIFEST="${RUN_DIR}/one.tsv"
{
  echo -e "wav_path\ttext"
  echo -e "${WAV_PATH}\t${GT_TEXT}"
} > "${MANIFEST}"

python "${SCRIPT_DIR}/make_npy_dataset.py" \
  --manifest-tsv "${MANIFEST}" \
  --out-dir "${RUN_DIR}/npy" \
  --out-list "${RUN_DIR}/one_list.txt" \
  --model-file "${MODEL_FILE}" \
  --time-frames 300

cp -f "$(head -n 1 "${RUN_DIR}/one_list.txt")" "${RUN_DIR}/input_0_float.npy"

echo "[2/6] float npy -> int8 input_0.dat"
python "${ROOT_DIR}/android13_t527_vpm_run/requantize_input_from_npy.py" \
  --float-npy "${RUN_DIR}/input_0_float.npy" \
  --meta "${META_PATH}" \
  --out-dat "${RUN_DIR}/input_0.dat"

echo "[3/6] make sample.txt"
cat > "${RUN_DIR}/sample.txt" <<EOF
[network]
${DEVICE_DIR}/network_binary.nb
[input]
${DEVICE_DIR}/input_0.dat
[output]
${DEVICE_DIR}/output_0.dat
EOF

echo "[4/6] adb push"
${ADB_BIN} shell "mkdir -p ${DEVICE_DIR}"
${ADB_BIN} push "${NB_PATH}" "${DEVICE_DIR}/network_binary.nb" >/dev/null
${ADB_BIN} push "${RUN_DIR}/input_0.dat" "${DEVICE_DIR}/input_0.dat" >/dev/null
${ADB_BIN} push "${RUN_DIR}/sample.txt" "${DEVICE_DIR}/sample.txt" >/dev/null

echo "[5/6] run NB on device"
${ADB_BIN} shell "cd ${DEVICE_DIR} && ${VPM_BIN_ON_DEVICE} -s sample.txt | tee ${DEVICE_DIR}/vpm_run.log"

echo "[6/6] pull output + decode"
${ADB_BIN} pull "${DEVICE_DIR}/output_0.dat" "${RUN_DIR}/output_0.dat" >/dev/null
${ADB_BIN} pull "${DEVICE_DIR}/vpm_run.log" "${RUN_DIR}/vpm_run.log" >/dev/null || true

python "${SCRIPT_DIR}/decode_nb_output_ko.py" \
  --dat "${RUN_DIR}/output_0.dat" \
  --meta "${META_PATH}" \
  --model-file "${MODEL_FILE}"

echo
echo "[DONE]"
echo " wav       : ${WAV_PATH}"
echo " run_dir   : ${RUN_DIR}"
echo " outputdat : ${RUN_DIR}/output_0.dat"
echo " vpm_log   : ${RUN_DIR}/vpm_run.log"


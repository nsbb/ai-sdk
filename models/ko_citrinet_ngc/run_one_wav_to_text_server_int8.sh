#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /abs/path/test.wav [GT_TEXT]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WAV_PATH="$(realpath "$1")"
GT_TEXT="${2:-}"

MODEL_FILE="${MODEL_FILE:-/nas02/geonhui83/stt/citrinet_korean/Citrinet-1024-gamma-0.25_spe-2048_ko-KR_Riva-ASR-SET-1.0.nemo}"
NEMO_ENV="${NEMO_ENV:-nemo_py310}"
ACUITY_ENV="${ACUITY_ENV:-acuitylite_py310}"
PEGASUS="${PEGASUS:-/nas02/geonhui83/T527_toolkit/acuity-toolkit-binary-6.12.0/bin/pegasus}"

MODEL_JSON="${MODEL_JSON:-${SCRIPT_DIR}/work/citrinet_npu.json}"
MODEL_DATA="${MODEL_DATA:-${SCRIPT_DIR}/work/citrinet_npu.data}"
MODEL_QUANT="${MODEL_QUANT:-${SCRIPT_DIR}/work/citrinet_npu_int8.quantize}"
INPUTMETA_TEMPLATE="${INPUTMETA_TEMPLATE:-${SCRIPT_DIR}/work/test_inputmeta.yml}"

RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/one_wav_server_int8}"
INFER_DIR="${RUN_DIR}/infer_out"
mkdir -p "${RUN_DIR}" "${INFER_DIR}" "${RUN_DIR}/npy"

if [[ ! -f "${WAV_PATH}" ]]; then
  echo "ERROR: wav not found: ${WAV_PATH}"
  exit 1
fi
if [[ ! -x "${PEGASUS}" ]]; then
  echo "ERROR: pegasus not found: ${PEGASUS}"
  exit 1
fi
for f in "${MODEL_FILE}" "${MODEL_JSON}" "${MODEL_DATA}" "${MODEL_QUANT}" "${INPUTMETA_TEMPLATE}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing file: ${f}"; exit 1; }
done

source /home1/Gunhee_Lee/anaconda3/etc/profile.d/conda.sh

echo "[1/5] wav -> npy feature"
conda activate "${NEMO_ENV}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER:-user}}"
mkdir -p "${NUMBA_CACHE_DIR}"

MANIFEST_TSV="${RUN_DIR}/one.tsv"
{
  echo -e "wav_path\ttext"
  echo -e "${WAV_PATH}\t${GT_TEXT}"
} > "${MANIFEST_TSV}"

python "${SCRIPT_DIR}/make_npy_dataset.py" \
  --manifest-tsv "${MANIFEST_TSV}" \
  --out-dir "${RUN_DIR}/npy" \
  --out-list "${RUN_DIR}/one_list.txt" \
  --model-file "${MODEL_FILE}" \
  --time-frames 300

META_TSV="${RUN_DIR}/one_list.meta.tsv"

echo "[2/5] make inputmeta for one wav"
ONE_INPUTMETA="${RUN_DIR}/one_inputmeta.yml"
python - <<PY
from pathlib import Path
p = Path("${INPUTMETA_TEMPLATE}")
out = Path("${ONE_INPUTMETA}")
target = str(Path("${RUN_DIR}/one_list.txt"))
lines = p.read_text().splitlines()
new = []
for line in lines:
    s = line.strip()
    indent = line[: len(line) - len(line.lstrip())]
    if s.startswith("path: "):
        new.append(f"{indent}path: {target}")
    elif s.startswith("- path: "):
        new.append(f"{indent}- path: {target}")
    else:
        new.append(line)
out.write_text("\\n".join(new) + "\\n")
print(f"[OK] {out}")
PY

echo "[3/5] pegasus INT8 inference (server)"
conda activate "${ACUITY_ENV}"
"${PEGASUS}" inference \
  --model "${MODEL_JSON}" \
  --model-data "${MODEL_DATA}" \
  --model-quantize "${MODEL_QUANT}" \
  --batch-size 1 \
  --iterations 1 \
  --device CPU \
  --with-input-meta "${ONE_INPUTMETA}" \
  --output-dir "${INFER_DIR}" \
  --dtype quantized \
  --postprocess dump_results

echo "[4/5] decode to text"
conda activate "${NEMO_ENV}"
OUT_TSV="${RUN_DIR}/decode_result.tsv"
python "${SCRIPT_DIR}/eval_test_cer.py" \
  --mode pegasus \
  --meta-tsv "${META_TSV}" \
  --model-file "${MODEL_FILE}" \
  --pegasus-out-dir "${INFER_DIR}" \
  --classes 2049 \
  --frames 38 \
  --out-tsv "${OUT_TSV}"

echo "[5/5] print result"
python - <<PY
import pandas as pd
df = pd.read_csv("${OUT_TSV}", sep="\\t")
row = df.iloc[0]
print("[WAV]", row["wav_path"])
print("[GT ]", row["gt"])
print("[PRED]", row["pred"])
PY

echo
echo "[DONE] run_dir=${RUN_DIR}"
echo " - infer tensor: ${INFER_DIR}"
echo " - decoded tsv : ${OUT_TSV}"


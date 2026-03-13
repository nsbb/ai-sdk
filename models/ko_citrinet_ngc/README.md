# Korean Citrinet NGC One-Shot Pipeline

This folder prepares Korean test/calibration data from
`/nas04/nlp_sk/STT/data/train`, then builds both:

- FP32 NB (`dtype=float32`)
- INT8 NB (`dtype=quantized`)

for T527 NPU.

## 1) Put model file

Place your downloaded NGC model as a `.nemo` file, for example:

`/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/model/speechtotext_ko_kr_citrinet_trainable_v1.0.nemo`

## 2) Run one-shot script

```bash
cd /nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc

bash run_pipeline_ko.sh \
  MODEL_NEMO=/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/model/speechtotext_ko_kr_citrinet_trainable_v1.0.nemo \
  TRAIN_CSV=/nas04/nlp_sk/STT/data/train/base/train_base_4356hr.csv \
  CALIB_COUNT=120 \
  TEST_COUNT=20 \
  QTYPE=int8 \
  ALGORITHM=moving_average \
  QUANTIZER=asymmetric_affine \
  MA_WEIGHT=0.004
```

If you want to use a model name instead of a local `.nemo`:

```bash
bash run_pipeline_ko.sh MODEL_NAME=<nemo_model_name>
```

## 3) Outputs

- `artifacts/int8/output_nbg_unify/network_binary.nb`
- `artifacts/int8/output_nbg_unify/nbg_meta.json`
- `artifacts/int8/output_nbg_unify/input_0.dat`
- `artifacts/fp32/output_nbg_unify/network_binary.nb`
- `artifacts/fp32/output_nbg_unify/nbg_meta.json`
- `artifacts/fp32/output_nbg_unify/input_0.dat`
- `bundle_int8/` and `bundle_fp32/` for `vpm_run`
- copied wavs and manifests under `data/`

## Notes

- Source train folder contains CSV manifests, not wav files directly.
- Wav paths are resolved from the CSV `raw_data` column.
- Data is copied (not symlinked) into this folder.

## Single Wav -> Text (INT8 NB)

Run one wav end-to-end on device (feature extraction + quantize input + vpm_run + decode):

```bash
cd /nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc

bash run_one_wav_to_text_int8.sh /abs/path/test.wav
```

Optional env overrides:

```bash
MODEL_FILE=/nas02/geonhui83/stt/citrinet_korean/Citrinet-1024-gamma-0.25_spe-2048_ko-KR_Riva-ASR-SET-1.0.nemo \
NB_PATH=/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/artifacts/int8/output_nbg_unify/network_binary.nb \
META_PATH=/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/artifacts/int8/output_nbg_unify/nbg_meta.json \
VPM_BIN_ON_DEVICE=/data/local/tmp/vpm_run_test/vpm_run \
bash run_one_wav_to_text_int8.sh /abs/path/test.wav "정답문자열(옵션)"
```

## Single Wav -> Text On Server (No Device)

If you want to test on server only (no adb/device), use pegasus inference:

```bash
cd /nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc
bash run_one_wav_to_text_server_int8.sh /abs/path/test.wav
```

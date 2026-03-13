# Korean Citrinet Handover (ONNX -> NB + 5s Input)

이 폴더는 인수인계용으로 다음을 한 번에 재현합니다.

1. ONNX 준비
- 로컬 ONNX 사용 (`ONNX_PATH`)
- URL 다운로드 (`ONNX_URL`)
- `.nemo`에서 직접 export

2. NB 생성
- INT8 NB
- FP32 NB

3. 실제 5초 입력 생성
- 입력 WAV를 정확히 5초로 trim/pad
- NeMo preprocessor로 feature 생성
- `input_5s_float.npy`, `input_5s.dat` 생성

## 폴더 구조

- `run_handover_pipeline.sh`: 원샷 파이프라인
- `make_5s_input.py`: 5초 입력만 별도로 생성
- `bundle_int8/`: 배포용 INT8 묶음
- `bundle_fp32/`: 배포용 FP32 묶음

## 필수 경로/환경

- Pegasus: `/nas02/geonhui83/T527_toolkit/acuity-toolkit-binary-6.12.0/bin/pegasus`
- NeMo conda env: `nemo_py310`
- 기본 모델: `/nas02/geonhui83/stt/citrinet_korean/Citrinet-1024-gamma-0.25_spe-2048_ko-KR_Riva-ASR-SET-1.0.nemo`

## 실행 예시

### A) `.nemo`에서 ONNX export + NB 생성 + 5초 입력 생성

```bash
cd /nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/handover_onnx_to_nb_5s
bash run_handover_pipeline.sh
```

### B) 로컬 ONNX 파일 사용

```bash
cd /nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/handover_onnx_to_nb_5s
ONNX_PATH=/abs/path/model.onnx bash run_handover_pipeline.sh
```

### C) ONNX URL 다운로드 후 사용

```bash
cd /nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/handover_onnx_to_nb_5s
ONNX_URL='https://.../model.onnx' bash run_handover_pipeline.sh
```

## 산출물

- `bundle_int8/network_binary.nb`
- `bundle_int8/nbg_meta.json`
- `bundle_int8/input_5s.dat`
- `bundle_int8/input_5s_float.npy`
- `bundle_int8/test_input_5s.wav`
- `bundle_int8/sample.txt`

- `bundle_fp32/network_binary.nb`
- `bundle_fp32/nbg_meta.json`
- `bundle_fp32/input_5s.dat`
- `bundle_fp32/test_input_5s.wav`
- `bundle_fp32/sample.txt`

## 주의

- 현재 기본 `TIME_FRAMES=500`입니다.
- 즉, 기본 실행은 입력 feature shape이 `[1,80,1,500]`인 5초 고정 NB를 생성합니다.
- 다른 길이를 쓰려면 `TIME_FRAMES`를 바꿔서 다시 빌드해야 합니다.

## 5초 입력만 별도 생성

기존 NB/메타를 그대로 쓰고 입력만 새로 만들 때:

```bash
source /home1/Gunhee_Lee/anaconda3/etc/profile.d/conda.sh
conda activate nemo_py310
python make_5s_input.py \
  --wav /abs/path/source.wav \
  --model-file /nas02/geonhui83/stt/citrinet_korean/Citrinet-1024-gamma-0.25_spe-2048_ko-KR_Riva-ASR-SET-1.0.nemo \
  --meta ./bundle_int8/nbg_meta.json \
  --out-wav ./bundle_int8/test_input_5s.wav \
  --out-float-npy ./bundle_int8/input_5s_float.npy \
  --out-dat ./bundle_int8/input_5s.dat \
  --duration-sec 5.0 \
  --time-frames 500
```

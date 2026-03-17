# Wav2Vec2 XLS-R 300M Korean — 한국어 (T527 NPU)

kresnik/wav2vec2-large-xlsr-korean (300M, 24L, 2617 음절). **8종+ 양자화 시도, 전부 실패.**

| 항목 | 값 |
|------|-----|
| 모델 | kresnik/wav2vec2-large-xlsr-korean |
| 입력 | `[1, 48000]` raw waveform (3초, 16kHz) |
| ONNX | 1.27GB |
| NB | 249MB (uint8) / 262MB (int16) |
| 추론시간 | 1,098ms (출력 ALL PAD) |
| Zeroth-Korean CER | **1.78%** (kresnik 공개 지표) |

## 상세 분석

- 양자화 실패 + 도메인 미스매치: [t527-stt/wav2vec2/xls-r-300m-korean/README.md](https://github.com/nsbb/t527-stt/wav2vec2/xls-r-300m-korean/)
- 전체 비교: [t527-stt/wav2vec2/README.md](https://github.com/nsbb/t527-stt/wav2vec2/)

## 주요 스크립트

| 파일 | 용도 |
|------|------|
| `make_fixed_onnx.py` | 동적 → 고정 shape ONNX |
| `prune_layers.py` | 24L → 12L pruning |
| `decode_npu_output.py` | NPU 출력 → 한국어 텍스트 |
| `prepare_calib_data.py` | calibration 데이터 생성 |
| `run_pipeline.sh` | 전체 변환 파이프라인 |
| `test_xlsr_opset12_nb.sh` | opset12 NB 디바이스 테스트 |
| `compare_onnx_npu.py` | ONNX vs NPU 비교 |

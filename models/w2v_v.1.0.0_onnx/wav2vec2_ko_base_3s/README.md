# Wav2Vec2 Base Korean — 한국어 (T527 NPU)

Kkonjeong/wav2vec2-base-korean (94.4M, 12L, 56 자모). **60종+ 양자화 시도, 전부 실패.**

| 항목 | 값 |
|------|-----|
| 모델 | Kkonjeong/wav2vec2-base-korean |
| 입력 | `[1, 48000]` raw waveform (3초, 16kHz) |
| NB | 72MB (uint8) / 153MB (int16) |
| 추론시간 | 425ms (동작하지만 출력 쓰레기) |
| ONNX FP32 CER | 9.5% (Zeroth-Korean) / 132~210% (월패드) |

## 상세 분석

- 양자화 실패 원인: [t527-stt/wav2vec2/base-korean/README.md](https://github.com/nsbb/t527-stt/wav2vec2/base-korean/)
- 전체 비교: [t527-stt/wav2vec2/README.md](https://github.com/nsbb/t527-stt/wav2vec2/)

## 주요 스크립트

| 파일 | 용도 |
|------|------|
| `download_and_convert.py` | HuggingFace → ONNX 변환 |
| `decode_ko_output.py` | NPU 출력 → 한국어 텍스트 |
| `prepare_ko_test_input.py` | 테스트 입력 생성 |
| `create_cnn_only_model.py` | CNN-only 분할 시도 |
| `smoothquant_v3.py` | SmoothQuant v3 (FP32 보존 성공, uint8 개선 없음) |
| `test_priority_nbs.sh` | 19종 우선순위 NPU 테스트 |
| `test_split_model.sh` | CNN(uint8) → Transformer(int16) 분리 추론 |
| `test_all_nbs.sh` | 전체 NB 일괄 테스트 |

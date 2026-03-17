# Wav2Vec2-base-960h 영어 STT (T527 NPU)

facebook/wav2vec2-base-960h. 영어 STT, T527 NPU uint8 양자화 **성공**.

| 항목 | 값 |
|------|-----|
| 모델 | facebook/wav2vec2-base-960h (94.4M, 12L Transformer) |
| 입력 | `[1, 80000]` raw waveform (5초, 16kHz) |
| NB 크기 | 87MB (uint8 asymmetric_affine) |
| CER | **17.52%** (ONNX FP32: 9.74%) |
| WER | **27.38%** |
| 추론시간 | **715ms** / 5초 (RTF 0.143) |
| 테스트셋 | LibriSpeech test-clean 50샘플 |

## 문서

- **[RESULTS.md](RESULTS.md)** — NPU 테스트 결과, 양자화 파라미터, CER/WER 상세, JNI 버그 수정 이력, 재현 방법
- **[acuity_612_vs_621.md](../../../docs/acuity_612_vs_621.md)** — Acuity 6.12 vs 6.21 양자화 비교 (6.12 uint8이 최적)

## 핵심 파일

| 파일 | 설명 |
|------|------|
| `wksp/wav2vec2_base_960h_5s_uint8_fixed_nbg_unify/network_binary.nb` | **최적 NB** (87MB, Acuity 6.12 uint8) |
| `wav2vec2_base_960h_5s_uint8_fixed.quantize` | 양자화 파일 |
| `wav2vec2_base_960h_5s_inputmeta.yml` | Acuity 입력 메타 (`reverse_channel: false` 필수) |
| `eval_wav2vec_cer.py` | CER/WER 평가 스크립트 |
| `data/english_test/` | 테스트 50샘플 + ground truth + NPU 출력 |

# Wav2Vec2 모델 변환 작업 디렉토리

facebook/wav2vec2 계열 모델의 T527 NPU 양자화 및 NB 변환 작업.

## 결과 요약

| 모델 | 언어 | CER | 추론시간 | NB | 상태 |
|------|------|-----|---------|-----|------|
| [wav2vec2_base_960h_5s/](wav2vec2_base_960h_5s/) | 영어 | **17.52%** | 715ms | 87MB | **성공** |
| [wav2vec2_ko_base_3s/](wav2vec2_ko_base_3s/) | 한국어 | — | 425ms | 72MB | 실패 (60종+ 시도) |
| [wav2vec2_ko_xls_r_300m_3s/](wav2vec2_ko_xls_r_300m_3s/) | 한국어 | — | 1,098ms | 249MB | 실패 (8종+ 시도) |

## 모델별 문서

| 폴더 | 설명 | 문서 |
|------|------|------|
| [wav2vec2_base_960h_5s/](wav2vec2_base_960h_5s/) | 영어 STT, uint8 성공 | [README](wav2vec2_base_960h_5s/README.md), [RESULTS](wav2vec2_base_960h_5s/RESULTS.md) |
| [wav2vec2_ko_base_3s/](wav2vec2_ko_base_3s/) | 한국어 base (94.4M, 12L, 56 자모) — 양자화 전부 실패 | |
| [wav2vec2_ko_xls_r_300m_3s/](wav2vec2_ko_xls_r_300m_3s/) | 한국어 XLS-R (300M, 24L, 2617 음절) — 양자화 전부 실패 | |

## 기타 폴더 (작업 잔재)

| 폴더 | 설명 |
|------|------|
| `0216/`, `0216_nbg_unify/` | 초기 변환 시도 |
| `wav2vec_original/`, `wav2vec_static/`, `wav2vec_5s/` | 초기 ONNX 변환 시도 |
| `wav2vec2_base_960h_5s_bf16/`, `_fp32/`, `_int16/`, `_uint8/`, `_fixed/` | 양자화 방식별 작업 폴더 |
| `wav2vec2_base_960h_20s/`, `_20s_bak/` | 20초 모델 시도 |
| `wav2ec2_base_960h_5s_fp32/` | 오타 포함된 fp32 시도 |
| `wav2vec_fixed_10s/` | 10초 모델 시도 |

## SDK 레벨 문서

- [Acuity 6.12 vs 6.21 비교](../../docs/acuity_612_vs_621.md) — 양자화 버전별 비교 (Wav2Vec2 영어 50샘플 실측)

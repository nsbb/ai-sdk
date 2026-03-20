# Wav2Vec2 모델 변환 작업 디렉토리

facebook/wav2vec2 계열 모델의 T527 NPU 양자화 및 NB 변환 작업.

## 결과 요약

| 모델 | 언어 | CER | 추론시간 | NB | 상태 |
|------|------|-----|---------|-----|------|
| [wav2vec2_base_960h_5s/](wav2vec2_base_960h_5s/) | 영어 | **17.52%** | 715ms | 87MB | **성공** |
| [wav2vec2_ko_eager_op12/](wav2vec2_ko_eager_op12/) | 한국어 | 측정 중 | 402ms | 67MB | **부분 성공** (non-blank 96.6%) |
| [wav2vec2_ko_base_3s/](wav2vec2_ko_base_3s/) | 한국어 | — | 425ms | 72MB | 실패 (60종+ 시도) |
| [wav2vec2_ko_xls_r_300m_3s/](wav2vec2_ko_xls_r_300m_3s/) | 한국어 | — | 1,098ms | 249MB | 실패 (8종+ 시도) |

## 한국어 모델 양자화 실험 결과 (2026-03-18)

**성공한 조합: Kkonjeong 원본 ONNX + PAD bias -8.0 + KL divergence**

상세 내용은 [wav2vec2_ko_eager_op12/QUANTIZATION_RESULTS.md](wav2vec2_ko_eager_op12/QUANTIZATION_RESULTS.md) 참조.

### 핵심 발견

1. **eager opset12 ONNX 재변환**이 결정적 — SDPA opset14 ONNX(667 nodes)에서는 46.3% agreement, eager opset12(957 nodes, EN과 동일)에서는 78.1%
2. **PAD bias 수정** (-8.0)이 non-blank preservation을 0.3% → 96.9%로 개선
3. **KL divergence** 알고리즘이 moving_average 대비 일관적으로 우수
4. **dropout=0.1 재학습은 실패** — non-blank agreement 60.2% → 29.7%로 악화 (시도하지 말 것)

### 다음 시도: int16 양자화

T527 NPU는 int16(dynamic_fixed_point)을 지원한다 — zipformer int16이 디바이스에서 실행 확인됨.
이전 wav2vec2 int16 "HANG" 보고는 VSIMULATOR_CONFIG 미설정(CID 불일치)이 원인일 가능성 높음.
uint8 CER ~133% garbled 문제를 int16로 해결할 수 있는 대안.

### 실패한 접근들 (시도하지 말 것)

| 접근 | 결과 | 이유 |
|------|------|------|
| dropout=0.1 재학습 후 PTQ | agreement 29.7% (악화) | CTC blank 과의존, PAD logit 증가 |
| SDPA ONNX + 다양한 calibration (12종) | max 46.3% | ONNX 구조 자체가 uint8 비호환 |
| XLS-R-300M (24L, 300M params) | 전부 실패 | 모델 크기 초과, attention 더 uniform |
| perchannel_symmetric_affine | 실패 | uint8에서 미지원 (int8만 가능) |
| MLE flag | 크래시 | Acuity 6.12 버그 |

## 모델별 문서

| 폴더 | 설명 | 문서 |
|------|------|------|
| [wav2vec2_base_960h_5s/](wav2vec2_base_960h_5s/) | 영어 STT, uint8 성공 | [README](wav2vec2_base_960h_5s/README.md), [RESULTS](wav2vec2_base_960h_5s/RESULTS.md) |
| [wav2vec2_ko_eager_op12/](wav2vec2_ko_eager_op12/) | 한국어 base, eager opset12 — **부분 성공** | [QUANTIZATION_RESULTS](wav2vec2_ko_eager_op12/QUANTIZATION_RESULTS.md) |
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

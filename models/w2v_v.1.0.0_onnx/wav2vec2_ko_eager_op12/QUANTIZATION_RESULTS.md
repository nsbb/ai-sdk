# wav2vec2-base-korean uint8 양자화 실험 결과

Kkonjeong/wav2vec2-base-korean (facebook/wav2vec2-base fine-tuned, 56 jamo CTC)을
T527 NPU uint8로 양자화하기 위한 전체 실험 기록.

## 최종 결과

**성공한 조합:** Kkonjeong 원본 eager opset12 ONNX + **PAD bias -8.0** + **KL divergence** calibration

| 지표 | 값 |
|------|-----|
| Non-blank preserved (시뮬) | 96.9% |
| Non-blank agreement (시뮬) | 60.2% |
| T527 NPU 추론 시간 | ~402ms |
| NB 파일 크기 | 67MB |
| Non-blank frames (디바이스) | 96.6% |
| CER (uint8 vs FP32) | ~133% (garbled) |
| CER (FP32 vs GT) | ~47% (3초 truncation 영향) |

> **참고:** FP32 모델 자체도 3초 truncation에서 CER ~47%. 원본 모델 공식 CER은 7.3% (전체 길이).

## ONNX 변환: eager opset12가 핵심

| ONNX 변환 방식 | 노드 수 | Opset | Overall agreement |
|---|---|---|---|
| SDPA + opset14 → onnxsim | 667 | 12 | 46.3% |
| **Eager + opset12 (native)** | **957** | **12** | **78.1%** |
| 영어 base-960h (참조) | 957 | 12 | — |

`export_eager_op12.py`로 `attn_implementation="eager"` 강제 → SDPA 비활성화 → opset 12 호환.
영어 모델과 100% 동일한 구조 (957 nodes).

## Calibration Sweep 결과 (20종)

### Phase 1: Kkonjeong 원본 ONNX 기반

| Variant | ONNX | Algorithm | Non-blank Preserved | Non-blank Agreement |
|---|---|---|---|---|
| v01_ma004_50 | orig | moving_average | 39.4% | 23.9% |
| v03_kl_50 | orig | kl_divergence | 50.1% | 28.9% |
| v11_padbias_ma004 | padbias -3 | moving_average | 75.7% | 33.1% |
| v12_padbias_kl | padbias -3 | kl_divergence | 82.5% | 40.7% |
| v17_padbias5_kl | padbias -5 | kl_divergence | 91.0% | 47.2% |
| **v18_padbias8_kl** | **padbias -8** | **kl_divergence** | **96.9%** | **60.2%** |

**패턴:**
- PAD bias 크기 ↑ → non-blank preserved ↑ (0→-3→-5→-8: 39%→82%→91%→97%)
- KL divergence > moving_average (일관적)
- Calibration 샘플 수 (50 vs 100): 차이 없음
- moving_average weight (0.004 vs 0.01 vs 0.1): 차이 없음

### Phase 2: dropout=0.1 재학습 (실패)

wav2vec2-base + attention_dropout=0.1 + hidden_dropout=0.1로 Zeroth-Korean CTC fine-tuning 후 양자화.

| Variant | ONNX | Algorithm | Non-blank Preserved | Non-blank Agreement |
|---|---|---|---|---|
| p2_kl50 | dropout=0.1 orig | kl_divergence | 11.2% | 5.8% |
| p2_pb8_kl50 | dropout=0.1 + padbias -8 | kl_divergence | 99.0% | **29.7%** |

**결론: dropout=0.1 재학습은 uint8 양자화를 악화시킨다.**

- FP32 eval CER: 8.57% (Kkonjeong 7.3%보다 나쁨)
- PAD logit 평균 8.97 (Kkonjeong보다 높음) → CTC blank 과의존
- non-blank agreement가 Kkonjeong 대비 절반 (29.7% vs 60.2%)
- 학습 환경: Docker pytorch/pytorch:2.4.1-cuda12.4, RTX 4070 Ti Super, 10 epoch 31분

## PAD bias 수정이 작동하는 원리

wav2vec2 CTC 모델에서 [PAD] (blank, id=53)의 logit이 비정상적으로 크다:
- Kkonjeong lm_head bias[53] = -9.98 (다른 토큰: -0.03 ~ +0.04)
- uint8 출력 범위: [-10, +12] → 22의 범위를 256 bins으로 커버
- PAD를 -8 내려서 [-2, +12] → 14 범위 → bin resolution ~40% 향상

```python
# PAD bias 수정 코드 (fix_lm_head_bias.py)
import onnx
from onnx import numpy_helper
m = onnx.load(input_onnx)
for init in m.graph.initializer:
    arr = numpy_helper.to_array(init).copy()
    if arr.shape == (56,):  # vocab size
        arr[53] += -8.0  # PAD token
        new_init = numpy_helper.from_array(arr, name=init.name)
        init.CopyFrom(new_init)
        break
onnx.save(m, output_onnx)
```

## NB Export 프로세스

1. Pegasus import: `pegasus import onnx --model model.onnx`
2. Pegasus quantize: `pegasus quantize --algorithm kl_divergence --quantizer asymmetric_affine --qtype uint8`
3. Pegasus export: `pegasus export ovxlib --with-input-meta inputmeta.yml` → VNN 소스 + .export.data 생성
4. Docker (t527-npu:v1.2)에서 gen_nbg 컴파일 + 실행:
   ```bash
   export VSIMULATOR_CONFIG="${VIV_SDK}/common/cfg/VIP9000NANOSI_PLUS_PID0X10000016.config"
   make -f makefile.linux
   ./gen_nbg model.export.data network_binary.nb
   ```
5. **주의:** VSIMULATOR_CONFIG 없으면 CID 0x10000020 (잘못됨) → 디바이스 로드 실패

## 디바이스 테스트

```bash
ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
$ADB push network_binary.nb input_0.dat sample.txt /data/local/tmp/wav2vec2_ko/
$ADB shell "cd /data/local/tmp && ./vpm_run_aarch64 -s wav2vec2_ko/sample.txt -b 0"
$ADB pull /data/local/tmp/wav2vec2_ko/output_0.dat .
```

Output dequantization: `float = (uint8 - 112) * 0.081849`, shape [56, 149, 1] → transpose → [149, 56]

## 관련 파일 위치

| 파일 | 위치 |
|---|---|
| 원본 ONNX | `wav2vec2_ko_eager_op12_3s.onnx` (이 디렉토리) |
| PAD bias 수정 ONNX | `/home/nsbb/travail/claude/T527/wav2vec2/work/onnx_padbias8/` |
| Calibration sweep 스크립트 | `/home/nsbb/travail/claude/T527/wav2vec2/work/run_calib_sweep.sh` |
| Sweep 결과 | `/home/nsbb/travail/claude/T527/wav2vec2/work/variants/` |
| NB 파일 | `/home/nsbb/travail/claude/T527/wav2vec2/work/export_workspace_v18_padbias8_kl/` |
| Phase 2 학습 (실패) | `/home/nsbb/travail/claude/T527/wav2vec2/work/phase2_train/` |
| 디코딩 스크립트 | `/home/nsbb/travail/claude/T527/wav2vec2/base-korean/decode_ko_output.py` |
| Vocab (56 jamo) | `vocab.json` (이 디렉토리) |

## int16 전체 양자화 (2026-03-20)

Acuity 6.21 + VivanteIDE 5.8.2 환경에서 전체 파이프라인 (import → quantize → inference → export) 실행.

### 시뮬레이션 결과

| 지표 | uint8 (v18 best) | int16 |
|------|-----------------|-------|
| Overall agreement | 67.1% | **98.0%** |
| Non-blank agreement | 60.2% | **98.8%** |
| Cosine similarity | 0.923 | **0.9998** |
| NB 크기 | 67MB | 152MB |

### 디바이스 결과

| NB 생성 환경 | 디바이스 | 추론시간 |
|-------------|---------|---------|
| Acuity 6.12 + VivanteIDE 5.7.2 | **HANG** | — |
| Acuity 6.21 + VivanteIDE 5.8.2 | 실행됨, **garbage** (NB agree 1.2%) | 1194ms |

- 5.7.2 HANG 원인: `layer_norm_axis0_I16_F32toI16_2D` VX 쉐이더 컴파일 실패
- 5.8.2 garbage 원인: NPU 하드웨어의 int16 Softmax/Erf 연산이 시뮬레이터와 불일치

### 레이어별 dump 분석 (FP32 vs int16 시뮬, 607개 레이어)

| Op 카테고리 | avg cos | min cos | 비고 |
|------------|---------|---------|------|
| InstanceNormalization | **1.000** | 1.000 | 완벽 |
| Conv (Feature Extractor) | **0.997** | 0.986 | 거의 완벽 |
| Softmax | 0.969 | **0.914** | 핵심 병목 |
| GELU (Erf+Mul) | 0.982 | **0.877** | 핵심 병목 |
| Residual Add | 0.971 | **0.783** | 오차 누적 |

**결론:** CNN/InstanceNorm은 int16에서 완벽. Transformer 내부의 **Softmax와 GELU(Erf)**가 int16 정밀도 부족의 핵심 원인. zipformer는 Erf 대신 Sigmoid를 사용하여 int16 동작.

### wav2vec2 vs zipformer op 비교

wav2vec2에만 있는 ops (zipformer에 없음):
- **Erf** (20개) — GELU activation. zipformer는 Sigmoid 사용
- **Sqrt** (26개) — LayerNorm
- **InstanceNormalization** (1개) — Feature Extractor GroupNorm

## hybrid 양자화 (uint8 + int16 혼합, 2026-03-20)

Pegasus `--hybrid` 플래그로 entropy 기반 레이어별 precision 지정.

| Variant | int16 레이어 | NB agree | 비고 |
|---------|-------------|----------|------|
| uint8 baseline (v18) | 0 | **58.0%** | — |
| CNN conv 3개 (v2) | 3 | 57.6% | OpenVINO 발견 재현 시도 |
| CNN conv 7개 (v3) | 7 | **58.6%** | +0.6%p 미미 |
| CNN+Softmax+PosConv+GELU (v1) | 45 | 46.3% | -11.7%p 악화 |

**결론:** Pegasus hybrid는 각 int16 레이어 경계에 dtype_converter(u8↔i16) 삽입 → 변환 오버헤드가 정밀도 이득을 상쇄. OpenVINO NNCF처럼 FP32로 복원하는 방식은 NPU에서 불가.

## OpenVINO NNCF 참조 (wav2vec2-base-960h)

OpenVINO의 NNCF PTQ with accuracy control:
- 단순 int8 전체: WER 0.947 → 0.500 (대폭 하락)
- accuracy-aware int8: WER 0.940 (0.7% 차이) — **feature extractor conv 3개를 FP32 복원**
- 94개 연산 중 3개만 FP32 → 정확도 거의 완전 복구

## entropy 기반 모델 분석

`entropy.txt` (Pegasus `--compute-entropy` 생성, 412개 레이어):

| 구간 | avg entropy | max entropy | 레이어 수 |
|------|------------|------------|----------|
| Feature Extractor CNN | 0.584 | **0.808** (conv1) | 26 |
| Transformer FFN (GELU) | 0.584 | **0.764** (layer 7) | 12 |
| Pos Conv Embed | 0.513 | 0.754 | 4 |
| Transformer Attention | 0.405 | **0.825** (Softmax layer 7) | 72 |

파라미터 분포: CNN 4.4% (4.2M), PosConv 5% (4.7M), Transformer 90% (85M), LM Head 0.05% (43K).

## TODO

- [ ] **GELU(Erf)→SiLU(Sigmoid) 대체 후 int16 재시도** — zipformer가 int16 되는 핵심 이유. `GELU(x) ≈ x * sigmoid(1.702*x)`
- [ ] Softmax+GELU만 uint8, 나머지 int16 hybrid 시도
- [ ] 모델 분할: CNN(int16) + Transformer(uint8)
- [ ] Zeroth-Korean test set에서 uint8 CER 실측
- [ ] CER 결과에 따라 QAT 또는 다른 모델 검토

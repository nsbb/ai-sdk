# Wav2Vec2 영어 모델: Acuity 6.12 vs 6.21 양자화 비교

Wav2Vec2-base-960h (94.4M params, 12-layer Transformer)를 T527 NPU에서 돌리기 위해
Acuity 6.12.0과 6.21.16 두 버전으로 양자화·NB 변환·디바이스 추론까지 비교한 결과.

테스트: LibriSpeech test-clean 50개 샘플, 5초 고정, 16kHz mono.

---

## 결과 요약

| 양자화 | Acuity | CER | WER | Exact | NB크기 | 추론시간 |
|--------|--------|-----|-----|-------|--------|---------|
| **uint8 asymmetric_affine** | **6.12** | **17.52%** | **27.38%** | **6/50** | **87MB** | **~720ms** |
| PCQ int8 perchannel_symmetric | 6.21 | 19.24% | 34.39% | 4/50 | 99MB | ~826ms |
| uint8 asymmetric_affine | 6.21 | 23.41% | 40.57% | 3/50 | 76MB | ~720ms |
| ONNX FP32 (reference) | — | 9.74% | — | — | — | — |

**결론: Acuity 6.12 uint8이 최적.** 6.21은 NB 크기가 작지만 정확도가 크게 떨어진다.

---

## 양자화 설정

### Acuity 6.12 uint8 (best)

```bash
# 로컬 WSL binary
source env.sh v3
pegasus quantize \
  --model wav2vec2_base_960h_5s.json \
  --model-data wav2vec2_base_960h_5s.data \
  --with-input-meta wav2vec2_base_960h_5s_inputmeta.yml \
  --device CPU \
  --quantizer asymmetric_affine --qtype uint8 \
  --rebuild-all \
  --algorithm moving_average --moving-average-weight 0.004 \
  --model-quantize wav2vec2_base_960h_5s_uint8_fixed.quantize
```

- Calibration: 51개 샘플 (LibriSpeech test-clean)
- `reverse_channel: false` 필수 (true로 하면 출력 blank)
- 입력: u8, scale=0.004272, zp=126
- 출력: u8, scale=0.15027, zp=186

### Acuity 6.21 PCQ int8

```bash
# Docker ubuntu-npu:v1.8.11 내부
PEGASUS='python3 /usr/local/acuity_command_line_tools/pegasus.py'
$PEGASUS quantize \
  --model wav2vec2_base_960h_5s.json \
  --model-data wav2vec2_base_960h_5s.data \
  --with-input-meta inputmeta_621.yml \
  --device CPU \
  --quantizer perchannel_symmetric_affine --qtype int8 \
  --rebuild-all \
  --algorithm moving_average --moving-average-weight 0.004 \
  --model-quantize wav2vec2_base_960h_5s_pcq_621.quantize
```

- Calibration: 51개 (동일)
- 입력: i8, scale=0.002860, zp=9
- 출력: i8, scale=0.15027, zp=58 (asymmetric_affine, per-tensor)

### Acuity 6.21 uint8

```bash
$PEGASUS quantize \
  --model wav2vec2_base_960h_5s.json \
  --model-data wav2vec2_base_960h_5s.data \
  --with-input-meta inputmeta_621.yml \
  --device CPU \
  --quantizer asymmetric_affine --qtype uint8 \
  --rebuild-all \
  --algorithm moving_average --moving-average-weight 0.004 \
  --model-quantize wav2vec2_base_960h_5s_uint8_621.quantize
```

- 입력: u8, scale=0.002860, zp=137
- 출력: u8, scale=0.15027, zp=186

---

## NB Export 환경

### Acuity 6.12 + VivanteIDE 5.7.2

```bash
# Docker t527-npu:v1.2
VSIM=/vivante57/cmdtools/vsimulator
export REAL_GCC=/usr/bin/gcc
export EXTRALFLAGS="-Wl,-rpath,$VSIM/lib -Wl,-rpath,$VSIM/lib/x64_linux/vsim"
cd /acuity612/bin
./pegasus export ovxlib \
  --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
  --viv-sdk $VSIM --target-ide-project linux64 --batch-size 1 ...
```

- OVXLIB 1.1.20
- `EXTRALFLAGS` 필수 (rpath)
- `cd /acuity612/bin` 필수 (template 경로)

### Acuity 6.21 + VivanteIDE 5.8.2

```bash
# Docker ubuntu-npu:v1.8.11
VSIM=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools/vsimulator
COMMON=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools/common/lib

# 핵심: VivanteIDE 라이브러리를 /usr/lib에 심링크
for lib in $VSIM/lib/*.so $COMMON/*.so; do
  ln -sf $lib /usr/lib/$(basename $lib) 2>/dev/null
done
ldconfig 2>/dev/null

$PEGASUS export ovxlib \
  --with-input-meta inputmeta_621.yml \
  --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
  --viv-sdk $VSIM --target-ide-project linux64 --batch-size 1 ...
```

- OVXLIB 1.2.18
- `--with-input-meta` 필수 (6.12에서는 export 시 불필요)
- libvdtproxy.so 심링크 필수

---

## 6.12 vs 6.21 주요 차이점

| 항목 | Acuity 6.12 | Acuity 6.21 |
|------|-------------|-------------|
| 설치 형태 | 바이너리 (standalone) | pip wheel (Python) |
| 실행 방법 | `./pegasus` | `python3 .../pegasus.py` |
| VivanteIDE 호환 | 5.7.2 (OVXLIB 1.1.20) | 5.8.2 (OVXLIB 1.2.18) |
| 양자화기 | asymmetric_affine, symmetric_affine, pcq, dfp, bf16, qbf16 | + **float16**, **e4m3**, **e5m2** |
| inputmeta lid | 관대 (mismatch 무시) | **엄격** (정확히 일치해야 함) |
| export 시 --with-input-meta | 불필요 | **필수** |
| NB 크기 (이 모델) | 87MB (uint8) | 76MB (uint8), 99MB (PCQ) |
| 정확도 (이 모델) | **CER 17.52%** | CER 19.24~23.41% |

---

## 왜 6.12가 더 좋은가

정확한 원인은 불명이나 추정:

1. **Graph optimization 차이**: 6.21은 더 공격적인 operator fusion을 수행 → NB 크기 감소(76MB vs 87MB)했지만, Transformer attention의 정밀도 손실 유발 가능
2. **양자화 calibration 구현 차이**: 동일 algorithm(moving_average)이라도 내부 구현이 다름
3. **PCQ가 이 모델에 맞지 않음**: per-channel scale이 attention 출력에서 불리 (vocab dim이 32밖에 안됨)

---

## 파일 위치

| 파일 | 설명 |
|------|------|
| `wksp/wav2vec2_base_960h_5s_uint8_fixed_nbg_unify/network_binary.nb` | **최적 NB** (87MB, Acuity 6.12) |
| `wksp_pcq_621_nbg_unify/network_binary.nb` | PCQ int8 NB (99MB, Acuity 6.21) |
| `wksp_uint8_621_nbg_unify_nbg_unify/network_binary.nb` | uint8 NB (76MB, Acuity 6.21) |
| `wav2vec2_base_960h_5s_uint8_fixed.quantize` | 6.12 uint8 quantize |
| `wav2vec2_base_960h_5s_pcq_621.quantize` | 6.21 PCQ quantize |
| `wav2vec2_base_960h_5s_uint8_621.quantize` | 6.21 uint8 quantize |
| `inputmeta_621.yml` | 6.21용 inputmeta (lid 수정) |
| `eval_wav2vec_cer.py` | CER/WER 평가 스크립트 |
| `data/english_test/` | 테스트 50샘플 + ground truth |
| `data/english_test/outputs/` | 6.12 uint8 NPU 출력 |
| `data/english_test/outputs_pcq_621/` | 6.21 PCQ NPU 출력 |
| `data/english_test/outputs_uint8_621/` | 6.21 uint8 NPU 출력 |

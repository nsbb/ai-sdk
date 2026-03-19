# Zipformer Streaming Korean STT — T527 NPU 양자화 실험

## 모델

**sherpa-onnx-streaming-zipformer-korean-2024-06-16** (k2-fsa/icefall)

| 컴포넌트 | 파일 | 크기 | 노드 수 |
|---|---|---|---|
| Encoder | encoder-epoch-99-avg-1.onnx | 280MB | 5868 |
| Decoder | decoder-epoch-99-avg-1.onnx | 2.8MB | — |
| Joiner | joiner-epoch-99-avg-1.onnx | 2.0MB | — |

- 아키텍처: RNN-T (Transducer), 5 stack Zipformer encoder
- Streaming: 39프레임(chunk) 단위, 30개 cached state 전달
- 토크나이저: BPE (tokens.txt, 500 vocab)

## 테스트셋

`test_wavs/` — 4개 한국어 WAV (KSS Dataset)

| 파일 | 텍스트 |
|---|---|
| 0.wav | 그는 괜찮은 척하려고 애쓰는 것 같았다. |
| 1.wav | 지하철에서 다리를 벌리고 앉지 마라. |
| 2.wav | 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다. |
| 3.wav | 주민등록증을 보여 주시겠어요? |

## ONNX 베이스라인

```
test_zipformer_onnx.py 실행 결과 (서버 CPU):
  전체 CER: 16.2%
  추론 시간: ~200ms/utterance
```

## T527 NPU 양자화 결과

**결론: 모든 양자화 방식 실패 (CER 100%)**

| 양자화 | NB 크기 | Encoder 출력 상관계수 | CER | 비고 |
|---|---|---|---|---|
| uint8 asymmetric_affine | 63MB | 0.627 | 100% | state input 수동 교정 |
| int16 dynamic_fixed_point | 118MB | 0.643 | 100% | state+내부 300개 노드 수동 교정 |
| PCQ int8 perchannel_symmetric | 71MB | 0.275 | 100% | 오히려 악화 |
| bf16 bfloat16 | — | — | — | export 실패 (error 64768) |

### 실패 원인

1. **양자화 에러 누적**: 5868개 노드의 sequential quantization으로 encoder 출력이 ONNX float 대비 상관계수 0.6 수준으로 열화. Decoder/Joiner가 의미있는 토큰을 생성하지 못함.

2. **Acuity multi-input 캘리브레이션 버그**: 31개 입력(mel + 30 state)을 사용하는 모델에서 state 입력의 캘리브레이션 데이터를 무시. 30개 state가 모두 scale=1.0/zp=0 (uint8) 또는 fl=300 (int16)으로 설정됨. 수동 교정이 필요했으나 내부 240개 관련 노드도 연쇄적으로 영향.

### 비교 (동일 T527 NPU)

| 모델 | 노드 수 | 구조 | uint8 CER | 비고 |
|---|---|---|---|---|
| KoCitrinet | ~200 | 1D Conv (CTC) | 44.44% | 성공 |
| Wav2Vec2 base | ~2000 | 12L Transformer (CTC) | ~25% | 성공 |
| **Zipformer** | **5868** | **5 stack Transformer (RNN-T)** | **100%** | **실패** |

## 파일 구조

```
zipformer/
├── README.md                          # 이 파일
├── test_zipformer_onnx.py             # ONNX 베이스라인 테스트
├── test_zipformer_npu_v2.py           # uint8 NPU 테스트 (state passing)
├── test_zipformer_npu_int16.py        # int16 NPU 테스트 (state passing)
├── compare_encoder_int16.py           # int16 vs ONNX 상관계수 비교
├── compare_encoder_pcq.py             # PCQ vs ONNX 상관계수 비교
└── zipformer_encoder_folded4/
    ├── encoder_with_states_v6_inputmeta.yml   # 최종 inputmeta (31 inputs)
    ├── fix_negative_gather_v2.py              # ONNX Gather 음수 인덱스 수정
    ├── fix_int16_quantize_all.py              # int16 quantize fl=300 수동 교정
    ├── fix_pcq_quantize_all.py                # PCQ quantize scale=1.0 수동 교정
    ├── *_uint8_v7.quantize                    # uint8 최종 quantize
    ├── *_int16_fixed_all.quantize             # int16 최종 quantize
    ├── *_pcq_fixed_all.quantize               # PCQ 최종 quantize
    ├── *_bf16.quantize                        # bf16 quantize (export 실패)
    └── dataset{0-30}.txt                      # 캘리브레이션 데이터 목록
```

## 재현 방법

```bash
# 1. ONNX 베이스라인
python3 test_zipformer_onnx.py

# 2. NPU 테스트 (T527 디바이스 연결 필요)
python3 test_zipformer_npu_int16.py
python3 compare_encoder_int16.py

# 3. NB 재생성 (Docker + Acuity 6.12.0 + VivanteIDE 5.7.2 필요)
# zipformer_encoder_folded4/ 디렉토리에서:
# import: pegasus import onnx --model *.onnx
# quantize: pegasus quantize --quantizer dynamic_fixed_point --qtype int16
# fix: python3 fix_int16_quantize_all.py
# export: pegasus export ovxlib --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016
```

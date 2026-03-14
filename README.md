# Allwinner T527 NPU AI SDK

Allwinner T527/T536 SoC에 내장된 **Vivante NPU**에서 AI 모델을 실행하기 위한 SDK입니다.
음성 인식(STT), 이미지 분류, 객체 탐지 등 다양한 모델을 NPU에서 추론할 수 있도록 변환·배포하는 전체 파이프라인을 포함합니다.

---

## 목차

1. [NPU란?](#npu란)
2. [전체 파이프라인](#전체-파이프라인)
3. [지원 플랫폼](#지원-플랫폼)
4. [디렉토리 구조](#디렉토리-구조)
5. [빠른 시작 — 예제 빌드](#빠른-시작--예제-빌드)
6. [모델 변환 파이프라인 (Pegasus)](#모델-변환-파이프라인-pegasus)
7. [탑재 모델 목록](#탑재-모델-목록)
8. [핵심 API](#핵심-api-awnn)
9. [T527 NPU 성능 벤치마크](#t527-npu-성능-벤치마크)
10. [환경 설정](#환경-설정)

---

## NPU란?

> **쉬운 설명**: CPU는 무엇이든 할 수 있는 만능 일꾼, GPU는 그림 작업에 특화된 일꾼, **NPU는 AI 연산(행렬 곱셈)만 전담하는 초고속 일꾼**입니다.
> AI 모델을 GPU나 CPU에서 돌리면 느리고 전기도 많이 쓰지만, NPU를 쓰면 빠르고 저전력입니다.

T527 NPU는 **Vivante VIP9000NanoDI Plus** 코어로, INT8 양자화 모델 기준 최대 **4 TOPS** 성능을 제공합니다.
NPU에서 모델을 실행하려면 `network_binary.nb`라는 전용 바이너리 포맷으로 변환해야 합니다.

---

## 전체 파이프라인

```
학습된 모델 (.nemo / .onnx / .tflite / .pb 등)
        │
        ▼  [1단계] Acuity Toolkit (Pegasus)
   Import → Quantize (INT8/FP16) → Export
        │
        ▼
  network_binary.nb  ◄── NPU 전용 바이너리
        │
        ├─ [2단계] 디바이스 검증
        │   vpm_run -s sample.txt
        │
        └─ [3단계] Android 앱 배포
            assets/models/ 에 nb + nbg_meta.json 탑재
            JNI (awnn_lib.c 또는 vnn_.c) 를 통해 NPU 호출
```

### 시스템 환경

```
Docker (NeMo/ONNX 변환)          WSL Ubuntu 20.04
  nemo_py310 conda env    →   /home/nsbb/travail/claude/T527/
  .nemo → .onnx                     │
                                    ▼
                          Acuity Toolkit (Pegasus)
                          .onnx → network_binary.nb
                                    │
                                    ▼
                          Android Studio (Windows)
                          APK 빌드 → adb → T527 보드
```

---

## 지원 플랫폼

| 플랫폼 | NPU 버전 | SW 드라이버 |
|--------|----------|-------------|
| **T527** | v2 | v1.13 (`libVIPlite.so`, `libVIPuser.so`) |
| T536 | v2 | v1.13 |
| MR527 | v2 | v1.13 |
| MR536 | v2 | v1.13 |
| T736 | v3 | v2.0 (`libNBGlinker.so`, `libVIPhal.so`) |
| A733 | v3 | v2.0 |
| AI985 | v3 | v2.0 |

플랫폼별 설정은 `machinfo/<platform>/config.mk` 참조.

---

## 디렉토리 구조

```
ai-sdk/
├── examples/                  # NPU 추론 예제 (C 소스)
│   ├── libawnn_viplite/       # ★ 핵심 NPU 래퍼 (awnn_lib.c, awnn_quantize.c)
│   ├── libawutils/            # 이미지 전처리 유틸
│   ├── resnet50/              # 이미지 분류 예제
│   ├── yolov5/                # 객체 탐지 예제
│   ├── deepspeech2/           # 음성 인식 예제
│   ├── multi_thread/          # 멀티스레드 병렬 추론 예제
│   └── vpm_run/               # 범용 NB 추론 도구
│
├── models/                    # 모델별 변환 스크립트 + 메타데이터
│   ├── env.sh                 # 환경 변수 설정 (NPU 버전 선택)
│   ├── pegasus_*.sh           # Pegasus 단계별 실행 스크립트
│   ├── ko_citrinet_ngc/       # ★ 한국어 음성 인식 (CitriNet)
│   ├── zipformer/             # 한국어 음성 인식 (Zipformer, 변환완료)
│   ├── w2v_v.1.0.0_onnx/      # 영어 음성 인식 (Wav2Vec2)
│   ├── CitriNet/              # 영어 음성 인식 (CitriNet)
│   ├── deepspeech2/           # 영어 음성 인식 (DeepSpeech2)
│   ├── MobileNetV2_Imagenet/  # 이미지 분류
│   ├── yolov5s-sim/           # 객체 탐지
│   └── ...                    # 기타 모델
│
├── machinfo/                  # 플랫폼별 컴파일 설정 (config.mk)
├── unified-tina/              # aarch64 Unified 드라이버 런타임 라이브러리
│   └── lib/aarch64-none-linux-gnu/  # libOpenVX.so, libovxlib.so, libGAL.so ...
└── viplite-tina/              # VIPLite 드라이버 라이브러리
```

---

## 빠른 시작 — 예제 빌드

### 사전 요구사항

- aarch64 크로스 컴파일러 (`aarch64-none-linux-gnu-gcc`)
- T527 보드 + adb 연결

### 빌드 및 설치

```bash
# T527용 빌드
AI_SDK_PLATFORM=t527 make

# 보드에 설치 (INSTALL_PREFIX 경로로 복사)
AI_SDK_PLATFORM=t527 make install
```

### adb로 보드에서 실행 (resnet50 예제)

```bash
adb push install/ /data/local/tmp/ai-sdk/
adb shell "cd /data/local/tmp/ai-sdk && ./resnet50/resnet50 resnet50/model/v2/resnet50.nb resnet50/input_data/cat.jpg"
```

---

## 모델 변환 파이프라인 (Pegasus)

> **쉬운 설명**: Pegasus는 일반 AI 모델을 NPU 전용 바이너리(`.nb`)로 변환해주는 도구입니다.
> "번역기"처럼 생각하면 됩니다 — PyTorch/TensorFlow 언어 → NPU 언어.

### 환경 설정

```bash
cd models/
source env.sh v3   # NPU 버전 설정 (T527/T536 = v3)

export ACUITY_PATH=/path/to/acuity-toolkit/bin/
export VIV_SDK=/path/to/VivanteIDE/cmdtools
```

### 원샷 변환 (권장)

```bash
pegasus_one MODEL_NAME          # 특정 모델 한 번에 변환
pegasus_auto                    # 전체 모델 일괄 변환
```

### 단계별 변환 (디버깅용)

```bash
# 1. Import: 원본 모델 → Acuity 내부 포맷
pegasus_import.sh MODEL_NAME

# 2. Quantize: FP32 → INT8 (정확도와 속도의 트레이드오프)
pegasus_quantize.sh MODEL_NAME uint8   # QType: uint8 / int16 / bf16

# 3. 정확도 검증 (선택)
pegasus_inference.sh MODEL_NAME float  # FP32 golden tensor 생성
pegasus_inference.sh MODEL_NAME uint8  # INT8 결과와 비교

# 4. Export: network_binary.nb 생성 ← 최종 NPU 실행 파일
pegasus_export_ovx_nbg.sh MODEL_NAME uint8
```

### 출력물

```
models/<MODEL_NAME>/wksp/<MODEL_NAME>_uint8/
└── output_nbg_unify/
    ├── network_binary.nb    ← NPU에서 실행할 바이너리
    └── nbg_meta.json        ← 양자화 파라미터 (scale, zero_point)
```

### 디바이스 검증 (vpm_run)

```bash
# sample.txt: 입력 데이터 경로 목록
./vpm_run -s sample.txt           # 1회 추론 + 출력 검증
./vpm_run -s sample.txt -l 100    # 100회 반복 (성능 측정)
```

### 변환 환경 (Docker 사용 시)

export ovxlib 단계에서 Linux gcc가 필요합니다. Docker 환경 권장:

```bash
docker run --rm \
  -v $WORK:/work \
  -v $VIVANTE57:/vivante57:ro \
  -v $ACUITY612:/acuity612:ro \
  t527-npu:v1.2 \
  bash -c "
    VSIM=/vivante57/cmdtools/vsimulator
    export REAL_GCC=/usr/bin/gcc
    export LD_LIBRARY_PATH=\$VSIM/lib:\$VSIM/lib/x64_linux/vsim:...
    export EXTRALFLAGS=\"-Wl,-rpath,\$VSIM/lib ...\"
    cd /acuity612/bin
    ./pegasus export ovxlib --pack-nbg-unify --viv-sdk \$VSIM ...
  "
```

> **핵심 주의사항**:
> - Acuity 6.12.0 + VivanteIDE 5.7.2 조합 필수 (버전 불일치 시 컴파일 오류)
> - `cd /acuity612/bin` 후 pegasus 실행 (vxcode/template 경로 문제 방지)
> - `REAL_GCC` 환경변수로 gcc 경로 명시 필요

---

## 탑재 모델 목록

### 음성 인식 (STT)

| 모델 | 언어 | 원본 프레임워크 | 입력 shape | NB 상태 | 비고 |
|------|------|----------------|------------|---------|------|
| **KoCitrinet (300f)** | 한국어 | NeMo → ONNX | `[1,80,1,300]` INT8 | ✅ 배포완료 | CER 44.4%, 120ms/frame |
| KoCitrinet (500f) | 한국어 | NeMo → ONNX | `[1,80,1,500]` INT8 | ✅ 변환완료 | Android 탑재 |
| CitriNet (EN, 3s) | 영어 | NeMo → ONNX | `[1,64,1,128]` | ✅ 변환완료 | FP32/INT8 |
| **Zipformer Encoder** | 한국어 | sherpa-onnx | `[1,39,80]` + 30 캐시 | ✅ 변환완료 (63MB) | 실기기 테스트 필요 |
| Zipformer Decoder | 한국어 | sherpa-onnx | `[1,512]` | ✅ 변환완료 (2.8MB) | |
| Zipformer Joiner | 한국어 | sherpa-onnx | `[1,512]` × 2 | ✅ 변환완료 (1.9MB) | |
| **Wav2Vec2 (5s)** | 영어 | HuggingFace → ONNX | `[1,80000]` INT8 | ✅ 추론성공 (87MB, 714ms) | |
| DeepSpeech2 | 영어 | TensorFlow | `[1,1,T,161]` | ✅ 변환완료 | |

### 이미지 분류 / 객체 탐지

| 모델 | 원본 프레임워크 | 입력 | NB 상태 |
|------|----------------|------|---------|
| MobileNet V1 (quant) | TFLite | `[1,224,224,3]` | ✅ |
| MobileNet V2 | Keras | `[1,224,224,3]` | ✅ |
| Inception V1 | ONNX | `[1,3,224,224]` | ✅ |
| SqueezeNet 1.0 | PyTorch | `[1,3,227,227]` | ✅ |
| LeNet | Caffe | `[1,1,28,28]` | ✅ |
| YOLOv5s | — | `[1,3,640,640]` | ✅ |
| YOLOv3-tiny | Darknet | `[1,3,416,416]` | ✅ |

---

## 핵심 API (awnn)

`examples/libawnn_viplite/awnn_lib.h` — C/C++ 양쪽에서 사용 가능

```c
#include "awnn_lib.h"

// 1. NPU 초기화 (프로세스당 1회)
awnn_init();

// 2. 모델 로드
Awnn_Context_t *ctx = awnn_create("model.nb");

// 3. 입력 데이터 설정 (양자화된 INT8 버퍼)
void *inputs[] = { input_buffer };
awnn_set_input_buffers(ctx, inputs);

// 4. 추론 실행
awnn_run(ctx);

// 5. 출력 획득
float **outputs = awnn_get_output_buffers(ctx);
// 또는: void *out = awnn_get_output_buffer(ctx, 0);

// 6. 정리
awnn_destroy(ctx);
awnn_uninit();
```

### 양자화 처리 (INT8 ↔ float32)

`nbg_meta.json`에 저장된 `scale` / `zero_point` 값을 사용합니다:

```c
// float → INT8 (입력 양자화)
int8_val = clamp(round(float_val / scale) + zero_point, -128, 127)

// INT8 → float (출력 역양자화)
float_val = (int8_val - zero_point) * scale
```

---

## T527 NPU 성능 벤치마크

측정 조건: T527 @ 546MHz / 696MHz, DRAM 1.2GHz, INT8 양자화

| 모델 | 입력 크기 | 546MHz FPS | 696MHz FPS |
|------|-----------|-----------|-----------|
| MobileNet V1 | 224² | 367 | — |
| MobileNet V2 | 224² | 309 | 381 |
| Inception V3 | 299² | 32.3 | 40.5 |
| SSD MobileNet V1 | 300² | 14.25 | 17.4 |
| YOLOv5s | 640² | 5.36 | 6.33 |
| YOLAct | 550² | 13.2 | 17.4 |
| **KoCitrinet (300f)** | `[1,80,1,300]` | **~8.3 FPS** (120ms) | — |
| **Wav2Vec2 (5s)** | `[1,80000]` | **1.4 FPS** (714ms) | — |

---

## 환경 설정

### 필수 툴체인

| 도구 | 버전 | 용도 |
|------|------|------|
| Acuity Toolkit | 6.12.0 | 모델 변환 (import/quantize/export) |
| VivanteIDE | 5.7.2 | NPU 시뮬레이터 (export 컴파일 시 필요) |
| NDK | r21+ | Android aarch64 바이너리 빌드 |
| aarch64-gcc | 10.3 | Linux aarch64 바이너리 빌드 |

### 드라이버 버전별 링크 플래그

**v1.13 (T527/T536)**:
```makefile
LDFLAGS += -lVIPlite -lVIPuser
```

**v2.0 (T736/A733)**:
```makefile
LDFLAGS += -lNBGlinker -lVIPhal
```

### 런타임 라이브러리 (디바이스에 필요)

`unified-tina/lib/aarch64-none-linux-gnu/` 내 `.so` 파일들을 디바이스 `/vendor/lib64/` 또는 `LD_LIBRARY_PATH`에 배치:

- `libOpenVX.so` — OpenVX 표준 API
- `libovxlib.so` — Vivante 확장 레이어
- `libGAL.so` — GPU Abstraction Layer (NPU 하드웨어 드라이버)
- `libVSC.so` — VeriSilicon Compiler
- `libNNVXCBinary.so` — VXC 커널 바이너리 (62MB)

---

## 참고 문서

- `docs/acuity_toolkit/` — Acuity Toolkit 사용자 가이드 (영문/중문)
- `docs/acuity_toolkit/NPU模块开发指南/` — NPU 모델 배포 전체 가이드 (중문)
- `docs/citrinet.pdf` — CitriNet 논문 (arXiv:2104.01721)
- `docs/wav2vec2/wav2vec2.pdf` — wav2vec 2.0 논문 (arXiv:2006.11477)

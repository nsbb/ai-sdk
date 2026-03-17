# Models

T527 NPU용 모델 변환 작업 디렉토리. ONNX/TF/Caffe → Acuity Pegasus → network_binary.nb

## Pegasus 파이프라인

```bash
source env.sh v3
pegasus_import.sh MODEL
pegasus_quantize.sh MODEL uint8
pegasus_export_ovx_nbg.sh MODEL uint8
```

## 모델 목록

### 음성 인식 (STT)

| 폴더 | 모델 | 언어 | 상태 | 문서 |
|------|------|------|------|------|
| [ko_citrinet_ngc/](ko_citrinet_ngc/) | KoCitrinet (NeMo) | 한국어 | **CER 44.44%, 120ms** | [README](ko_citrinet_ngc/README.md) |
| [w2v_v.1.0.0_onnx/](w2v_v.1.0.0_onnx/) | Wav2Vec2 (영어/한국어) | 영어+한국어 | 영어 CER 17.52%, 한국어 실패 | [README](w2v_v.1.0.0_onnx/README.md) |
| [zipformer/](zipformer/) | Zipformer (sherpa-onnx) | 한국어 | NB 변환 완료, 테스트 대기 | |
| [deepspeech2/](deepspeech2/) | DeepSpeech2 (TF) | 영어 | NB 변환 완료 | |
| [CitriNet/](CitriNet/) | CitriNet (NeMo) | 영어 | NB 변환 완료 | |

### 이미지 분류 / 객체 탐지

| 폴더 | 모델 | 상태 |
|------|------|------|
| [MobileNetV2_Imagenet/](MobileNetV2_Imagenet/) | MobileNet V2 (Keras) | 변환 완료 |
| [mobilenet_v1_1.0_224_quant/](mobilenet_v1_1.0_224_quant/) | MobileNet V1 (TFLite, 이미 양자화) | 변환 완료 |
| [inception_v1/](inception_v1/) | Inception V1 (ONNX) | 변환 완료 |
| [resnet50-sim/](resnet50-sim/) | ResNet50 | 변환 완료 |
| [squeezenet1_0/](squeezenet1_0/) | SqueezeNet (PyTorch) | 변환 완료 |
| [lenet/](lenet/) | LeNet (Caffe) | 변환 완료 |
| [yolov5s-sim/](yolov5s-sim/) | YOLOv5s | 변환 완료 |
| [yolov3_tiny/](yolov3_tiny/) | YOLOv3 Tiny (Darknet) | 변환 완료 |

### 기타

| 폴더 | 모델 | 상태 |
|------|------|------|
| [lstm_mnist/](lstm_mnist/) | LSTM MNIST (TF) | 변환 완료 |

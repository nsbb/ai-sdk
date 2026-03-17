#!/usr/bin/env python3
"""
ONNX Runtime static uint8 양자화로 원본 vs activation-clamped 모델 비교.
Pegasus/NPU 양자화를 근사 시뮬레이션.
"""
import os
import glob
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantFormat, QuantType
)
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "aug_calib_npy")


class Wav2VecCalibReader(CalibrationDataReader):
    def __init__(self, calib_dir, input_name, num_samples=30):
        self.files = sorted(glob.glob(os.path.join(calib_dir, "*.npy")))[:num_samples]
        self.input_name = input_name
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        data = np.load(self.files[self.idx]).astype(np.float32)
        if data.shape != (1, 48000):
            data = data.reshape(1, -1)[:, :48000]
            if data.shape[1] < 48000:
                data = np.pad(data, ((0, 0), (0, 48000 - data.shape[1])))
        self.idx += 1
        return {self.input_name: data}


def quantize_model(model_path, label, num_calib=30):
    """ONNX Runtime static uint8 양자화"""
    print(f"\n{'='*70}")
    print(f"Quantizing: {label}")
    print(f"{'='*70}")

    out_path = model_path.replace(".onnx", "_ort_uint8.onnx")

    # 입력 이름 확인
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    reader = Wav2VecCalibReader(CALIB_DIR, input_name, num_samples=num_calib)

    try:
        quantize_static(
            model_input=model_path,
            model_output=out_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QOperator,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
            per_channel=False,  # per-tensor (NPU와 동일)
        )
        print(f"  Quantized model saved: {os.path.basename(out_path)}")
        print(f"  Size: {os.path.getsize(out_path)/1e6:.1f}MB")
        return out_path
    except Exception as e:
        print(f"  Quantization failed: {e}")
        return None


def evaluate(model_path, test_input, label):
    """모델 평가"""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: test_input})[0]

    logit_range = output.max() - output.min()
    top2 = np.sort(output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]

    vocab_path = os.path.join(BASE_DIR, "vocab.json")
    with open(vocab_path) as f:
        vocab_dict = json.load(f)
    vocab = [""] * (max(vocab_dict.values()) + 1)
    for char, idx in vocab_dict.items():
        vocab[idx] = char

    tokens = np.argmax(output[0], axis=-1)
    prev = -1
    decoded = []
    for t in tokens:
        if t != prev:
            if t != 0:
                decoded.append(t)
            prev = t
    text = "".join(vocab[t] for t in decoded if t < len(vocab))

    print(f"  [{label}]")
    print(f"    Logit range: {logit_range:.2f}, Top1-2 gap: {gaps.mean():.3f}")
    print(f"    Decoded: {text}")
    return output, text


def compare_outputs(fp32_output, quant_output, label):
    """FP32 vs Quantized 출력 비교"""
    # argmax 일치율
    fp32_tokens = np.argmax(fp32_output[0], axis=-1)
    quant_tokens = np.argmax(quant_output[0], axis=-1)
    agreement = np.mean(fp32_tokens == quant_tokens)

    # logit 차이
    diff = np.abs(fp32_output - quant_output)
    mean_diff = diff.mean()
    max_diff = diff.max()

    print(f"  [{label} FP32↔Quant comparison]")
    print(f"    Argmax agreement: {agreement*100:.1f}%")
    print(f"    Logit diff: mean={mean_diff:.4f}, max={max_diff:.4f}")


def main():
    test_input = np.load(os.path.join(BASE_DIR, "test_audio.npy"))

    models = {
        "Original": os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx"),
        "Clip50": os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_clip50_nopad10_opset12_sim.onnx"),
        "Clip30": os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_clip30_nopad10_opset12_sim.onnx"),
    }

    results = {}

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"\n{name}: {path} not found, skipping")
            continue

        # FP32 평가
        print(f"\n{'='*70}")
        print(f"Model: {name}")
        fp32_out, fp32_text = evaluate(path, test_input, f"{name} FP32")

        # uint8 양자화
        quant_path = quantize_model(path, name, num_calib=30)
        if quant_path:
            quant_out, quant_text = evaluate(quant_path, test_input, f"{name} uint8")
            compare_outputs(fp32_out, quant_out, name)
            results[name] = {
                'fp32_text': fp32_text,
                'quant_text': quant_text,
                'fp32_range': fp32_out.max() - fp32_out.min(),
                'quant_range': quant_out.max() - quant_out.min(),
            }

    # 요약
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<12} {'FP32 Range':>12} {'Quant Range':>12} {'FP32 Text':<30} {'Quant Text':<30}")
        print("-" * 100)
        for name, r in results.items():
            fp32_short = r['fp32_text'][:28]
            quant_short = r['quant_text'][:28]
            print(f"  {name:<10} {r['fp32_range']:>10.2f}   {r['quant_range']:>10.2f}   "
                  f"{fp32_short:<28}   {quant_short:<28}")


if __name__ == "__main__":
    main()

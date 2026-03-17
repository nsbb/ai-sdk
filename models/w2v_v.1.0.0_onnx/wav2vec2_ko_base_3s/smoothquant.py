#!/usr/bin/env python3
"""
SmoothQuant for Korean wav2vec2.

핵심 아이디어: activation 동적 범위를 weight로 이전하여 양자화 품질 개선.
per-channel smoothing factor s_j = max(|X_j|)^α / (max(|W_j|)^(1-α))
X_new = X / diag(s), W_new = diag(s) @ W

wav2vec2 구조:
  LayerNorm → Q/K/V projection (attention)
  LayerNorm → FC1/FC2 (FFN)

LayerNorm의 scale(γ)에 smoothing factor를 흡수시키면
모델 구조 변경 없이 activation range 축소 가능.
"""
import os
import glob
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
import json
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KO_ONNX = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx")
CALIB_DIR = os.path.join(BASE_DIR, "aug_calib_npy")


def get_all_linear_io(model):
    """
    ONNX 그래프에서 모든 Linear (MatMul → Add) 패턴을 찾고,
    각 Linear의 입력 텐서 이름, weight 이름, bias 이름을 반환.
    """
    init_names = {init.name for init in model.graph.initializer}
    init_shapes = {}
    for init in model.graph.initializer:
        init_shapes[init.name] = list(init.dims)

    # MatMul 노드에서 weight (initializer)와 입력을 찾기
    linears = []
    for node in model.graph.node:
        if node.op_type == "MatMul":
            weight_name = None
            input_name = None
            for inp in node.input:
                if inp in init_names and len(init_shapes.get(inp, [])) == 2:
                    weight_name = inp
                else:
                    input_name = inp

            if weight_name and input_name:
                # 뒤에 Add (bias)가 있는지 찾기
                bias_name = None
                for node2 in model.graph.node:
                    if node2.op_type == "Add" and node.output[0] in node2.input:
                        for inp in node2.input:
                            if inp in init_names and inp != node.output[0]:
                                bias_name = inp
                        break

                linears.append({
                    'matmul_node': node.name,
                    'input': input_name,
                    'weight': weight_name,
                    'weight_shape': init_shapes[weight_name],
                    'bias': bias_name,
                    'output': node.output[0],
                })

    return linears


def find_preceding_layernorm(model, linear_input_name):
    """
    Linear의 입력이 어떤 LayerNorm의 출력인지 추적.
    Returns: (layernorm_weight_name, layernorm_bias_name) or None
    """
    # linear_input_name을 output으로 가진 노드 찾기
    for node in model.graph.node:
        if linear_input_name in node.output:
            # LayerNorm은 ONNX에서 여러 노드로 분해됨
            # 보통: ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add
            # 마지막 Mul이 γ, 마지막 Add가 β
            # 또는 명시적 LayerNormalization 노드
            if node.op_type == "Add":
                # Mul → Add 패턴의 Add (β)
                # Mul을 찾기
                for inp in node.input:
                    for node2 in model.graph.node:
                        if node2.op_type == "Mul" and inp in node2.output:
                            # 이 Mul의 initializer 입력이 γ
                            init_names = {init.name for init in model.graph.initializer}
                            for inp2 in node2.input:
                                if inp2 in init_names:
                                    # γ 찾음
                                    gamma_name = inp2
                                    # Add의 initializer가 β
                                    for inp3 in node.input:
                                        if inp3 in init_names:
                                            return gamma_name, inp3
            return None
    return None


def collect_activation_stats(onnx_path, calib_dir, num_samples=50):
    """calibration 데이터로 per-channel activation max abs 수집"""
    print(f"\nCollecting activation statistics from {num_samples} calibration samples...")

    model = onnx.load(onnx_path)
    linears = get_all_linear_io(model)
    print(f"  Found {len(linears)} linear layers")

    # 각 linear의 입력 텐서를 graph output에 추가
    existing_outputs = {o.name for o in model.graph.output}
    shape_info = {vi.name: vi for vi in model.graph.value_info}

    linear_input_names = list(set(l['input'] for l in linears))
    for name in linear_input_names:
        if name not in existing_outputs:
            if name in shape_info:
                model.graph.output.append(shape_info[name])
            else:
                from onnx import TensorProto
                model.graph.output.append(
                    onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                )

    # 임시 모델 저장
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name
        onnx.save(model, tmp_path)

    try:
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_names = [o.name for o in sess.get_outputs()]

        # per-channel max abs 누적
        channel_max = {}  # name → max_abs per channel

        calib_files = sorted(glob.glob(os.path.join(calib_dir, "*.npy")))[:num_samples]
        print(f"  Using {len(calib_files)} calibration files")

        for i, calib_file in enumerate(calib_files):
            data = np.load(calib_file)
            if data.shape != (1, 48000):
                data = data.reshape(1, -1)[:, :48000]
                if data.shape[1] < 48000:
                    data = np.pad(data, ((0, 0), (0, 48000 - data.shape[1])))

            results = sess.run(output_names, {input_name: data.astype(np.float32)})

            for name, result in zip(output_names, results):
                if name in linear_input_names and isinstance(result, np.ndarray):
                    # result shape: [batch, seq_len, hidden_dim] 또는 [batch, hidden_dim]
                    if result.ndim == 3:
                        # per-channel max abs (마지막 차원이 channel)
                        per_ch = np.max(np.abs(result), axis=(0, 1))
                    elif result.ndim == 2:
                        per_ch = np.max(np.abs(result), axis=0)
                    else:
                        continue

                    if name not in channel_max:
                        channel_max[name] = per_ch
                    else:
                        channel_max[name] = np.maximum(channel_max[name], per_ch)

            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(calib_files)}")

    finally:
        os.unlink(tmp_path)

    print(f"  Collected stats for {len(channel_max)} activation tensors")
    return channel_max, linears


def apply_smoothquant(onnx_path, channel_max, linears, alpha=0.5):
    """
    SmoothQuant 적용.
    s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
    Then: X_new = X / s, W_new = s * W
    """
    print(f"\n{'='*70}")
    print(f"Applying SmoothQuant (alpha={alpha})")
    print(f"{'='*70}")

    model = onnx.load(onnx_path)
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    smoothed = 0
    total_range_reduction = []

    for linear in linears:
        input_name = linear['input']
        weight_name = linear['weight']
        weight_shape = linear['weight_shape']

        if input_name not in channel_max:
            continue

        act_max = channel_max[input_name]  # per-channel activation max abs
        weight = init_map.get(weight_name)
        if weight is None:
            continue

        # weight shape: [out_dim, in_dim] for nn.Linear
        # per-channel weight max abs (input channel dimension)
        if weight.shape[1] == len(act_max):
            # weight: [out, in], act_max: [in]
            weight_max = np.max(np.abs(weight), axis=0)  # [in]
        elif weight.shape[0] == len(act_max):
            # weight: [in, out] (transposed)
            weight_max = np.max(np.abs(weight), axis=1)  # [in]
        else:
            continue

        # smoothing factor
        act_max_safe = np.clip(act_max, 1e-5, None)
        weight_max_safe = np.clip(weight_max, 1e-5, None)

        s = np.power(act_max_safe, alpha) / np.power(weight_max_safe, 1 - alpha)
        s = np.clip(s, 1e-5, 1e5)

        # Apply: W_new = diag(s) @ W (scale input channels of weight)
        if weight.shape[1] == len(act_max):
            new_weight = weight * s[np.newaxis, :]  # broadcast [out, in] * [in]
        else:
            new_weight = weight * s[:, np.newaxis]  # broadcast [in, out] * [in, 1]

        # 범위 변화 추적
        old_w_range = weight.max() - weight.min()
        new_w_range = new_weight.max() - new_weight.min()

        # LayerNorm의 γ에 1/s 흡수 (activation 축소)
        ln_info = find_preceding_layernorm(model, input_name)

        if ln_info:
            gamma_name, beta_name = ln_info
            gamma = init_map.get(gamma_name)
            if gamma is not None and len(gamma) == len(s):
                new_gamma = gamma / s
                # gamma 업데이트
                for init in model.graph.initializer:
                    if init.name == gamma_name:
                        new_t = numpy_helper.from_array(new_gamma.astype(np.float32), gamma_name)
                        init.CopyFrom(new_t)
                        break
                init_map[gamma_name] = new_gamma

        # weight 업데이트
        for init in model.graph.initializer:
            if init.name == weight_name:
                new_t = numpy_helper.from_array(new_weight.astype(np.float32), weight_name)
                init.CopyFrom(new_t)
                break
        init_map[weight_name] = new_weight

        smoothed += 1
        total_range_reduction.append(new_w_range / (old_w_range + 1e-10))

    print(f"  Smoothed {smoothed}/{len(linears)} linear layers")
    if total_range_reduction:
        print(f"  Avg weight range change: {np.mean(total_range_reduction):.3f}x")

    return model


def evaluate_model(onnx_path_or_model, label, test_input):
    """모델 평가"""
    if isinstance(onnx_path_or_model, str):
        path = onnx_path_or_model
    else:
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name
            onnx.save(onnx_path_or_model, path)

    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: test_input})[0]
    finally:
        if not isinstance(onnx_path_or_model, str):
            os.unlink(path)

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

    return output


def simulate_quantized_inference(onnx_path, test_input, label):
    """uint8 양자화 시뮬레이션 — 모든 activation과 weight를 uint8로 양자화한 후 추론"""
    model = onnx.load(onnx_path)

    # Weight를 uint8로 양자화
    for init in model.graph.initializer:
        data = numpy_helper.to_array(init)
        if data.size < 100:
            continue
        fmin, fmax = data.min(), data.max()
        if fmax == fmin:
            continue
        scale = (fmax - fmin) / 255.0
        zp = int(np.round(-fmin / scale))
        zp = np.clip(zp, 0, 255)
        quantized = np.clip(np.round(data / scale + zp), 0, 255).astype(np.uint8)
        dequantized = (quantized.astype(np.float32) - zp) * scale

        new_t = numpy_helper.from_array(dequantized, init.name)
        init.CopyFrom(new_t)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name
        onnx.save(model, tmp_path)

    try:
        output = evaluate_model(tmp_path, f"{label} (weight-quant sim)", test_input)
    finally:
        os.unlink(tmp_path)

    return output


def main():
    print("=" * 70)
    print("SmoothQuant for Korean Wav2Vec2")
    print("=" * 70)

    test_input = np.load(os.path.join(BASE_DIR, "test_audio.npy"))
    print(f"Test input: {test_input.shape}")

    # 원본 평가
    print("\n--- Original Model ---")
    orig_output = evaluate_model(KO_ONNX, "Original FP32", test_input)

    # 원본 weight 양자화 시뮬레이션
    print("\n--- Original Weight Quantization Simulation ---")
    simulate_quantized_inference(KO_ONNX, test_input, "Original")

    # Activation 통계 수집
    channel_max, linears = collect_activation_stats(KO_ONNX, CALIB_DIR, num_samples=50)

    # 여러 alpha 값으로 SmoothQuant 시도
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        model = apply_smoothquant(KO_ONNX, channel_max, linears, alpha=alpha)

        out_path = os.path.join(BASE_DIR,
                                f"wav2vec2_ko_base_3s_smooth_a{alpha}_nopad10_opset12_sim.onnx")
        onnx.save(model, out_path)
        print(f"  Saved: {os.path.basename(out_path)} ({os.path.getsize(out_path)/1e6:.1f}MB)")

        # FP32 평가
        evaluate_model(out_path, f"SmoothQuant α={alpha} FP32", test_input)

        # Weight 양자화 시뮬레이션
        simulate_quantized_inference(out_path, test_input, f"SmoothQuant α={alpha}")

    print("\n" + "=" * 70)
    print("DONE — SmoothQuant models ready for Pegasus quantization")
    print("=" * 70)


if __name__ == "__main__":
    main()

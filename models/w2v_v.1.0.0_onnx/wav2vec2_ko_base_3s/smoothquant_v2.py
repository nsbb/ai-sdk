#!/usr/bin/env python3
"""
SmoothQuant v2 for Korean wav2vec2.

v1 대비 수정:
1. LayerNorm β도 /s 적용 (v1은 γ만 수정 → FP32 파괴)
2. Q/K/V 공유 입력 그룹 처리 (v1은 같은 γ에 s를 3번 적용)
3. LayerNorm 추적 개선 (Mul→Add 패턴 정확 매칭)
4. FC2/out_proj 제외 (LayerNorm 직접 연결 없음)

SmoothQuant 수학:
  Y = X @ W = (X diag(s)^{-1}) @ (diag(s) W)
  X는 LayerNorm 출력: x = γ * norm(z) + β
  X' = X / s = (γ/s) * norm(z) + (β/s)
  → γ_new = γ/s, β_new = β/s, W_new = diag(s) @ W
  FP32에서 결과는 수학적으로 동일해야 함.
"""
import os
import glob
import copy
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, TensorProto
import json
import tempfile
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KO_ONNX = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx")
CALIB_DIR = os.path.join(BASE_DIR, "aug_calib_npy")


def build_graph_maps(model):
    """ONNX 그래프의 연결 맵 구축"""
    init_names = {init.name for init in model.graph.initializer}
    init_shapes = {}
    for init in model.graph.initializer:
        init_shapes[init.name] = list(init.dims)

    output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    return init_names, init_shapes, output_to_node


def find_linear_groups(model):
    """
    Linear 레이어를 입력 텐서로 그룹화.
    Q/K/V는 같은 입력 → 같은 그룹.
    Returns: {input_name: [{'weight': ..., 'bias': ..., 'matmul': ...}, ...]}
    """
    init_names, init_shapes, output_to_node = build_graph_maps(model)

    groups = defaultdict(list)
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue

        weight_name = None
        input_name = None
        for inp in node.input:
            if inp in init_names and len(init_shapes.get(inp, [])) == 2:
                weight_name = inp
            else:
                input_name = inp

        if not weight_name or not input_name:
            continue

        # bias (Add 뒤에 있는지)
        bias_name = None
        for node2 in model.graph.node:
            if node2.op_type == "Add" and node.output[0] in node2.input:
                for inp in node2.input:
                    if inp in init_names and inp != node.output[0]:
                        bias_name = inp
                break

        groups[input_name].append({
            'matmul_name': node.name,
            'weight': weight_name,
            'weight_shape': init_shapes[weight_name],
            'bias': bias_name,
        })

    return groups


def find_layernorm_params(model, linear_input_name):
    """
    linear_input_name을 생성하는 LayerNorm의 γ, β initializer 이름을 찾는다.

    패턴: ... → Div → Mul(γ) → Add(β) → linear_input_name
    β = Add의 initializer, γ = Mul의 initializer
    """
    init_names = {init.name for init in model.graph.initializer}
    _, _, output_to_node = build_graph_maps(model)

    if linear_input_name not in output_to_node:
        return None

    # linear_input_name을 출력하는 Add 노드 (β)
    add_node = output_to_node[linear_input_name]
    if add_node.op_type != "Add":
        return None

    # Add의 initializer = β
    beta_name = None
    mul_output = None
    for inp in add_node.input:
        if inp in init_names:
            beta_name = inp
        elif inp in output_to_node:
            mul_output = inp

    if not beta_name or not mul_output:
        return None

    # Mul 노드 (γ)
    mul_node = output_to_node.get(mul_output)
    if mul_node is None or mul_node.op_type != "Mul":
        return None

    # Mul의 initializer = γ
    gamma_name = None
    for inp in mul_node.input:
        if inp in init_names:
            gamma_name = inp

    if not gamma_name:
        return None

    return gamma_name, beta_name


def collect_activation_stats(onnx_path, calib_dir, num_samples=50):
    """calibration 데이터로 per-channel activation max abs 수집"""
    print(f"\nCollecting activation statistics from {num_samples} samples...")

    model = onnx.load(onnx_path)
    groups = find_linear_groups(model)

    # 각 그룹의 입력 텐서를 graph output에 추가
    existing_outputs = {o.name for o in model.graph.output}
    shape_info = {vi.name: vi for vi in model.graph.value_info}

    input_names = list(groups.keys())
    for name in input_names:
        if name not in existing_outputs:
            if name in shape_info:
                model.graph.output.append(shape_info[name])
            else:
                model.graph.output.append(
                    onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, None))

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name
        onnx.save(model, tmp_path)

    try:
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_names = [o.name for o in sess.get_outputs()]

        channel_max = {}

        calib_files = sorted(glob.glob(os.path.join(calib_dir, "*.npy")))[:num_samples]
        print(f"  Using {len(calib_files)} calibration files")

        for i, calib_file in enumerate(calib_files):
            data = np.load(calib_file).astype(np.float32)
            if data.shape != (1, 48000):
                data = data.reshape(1, -1)[:, :48000]
                if data.shape[1] < 48000:
                    data = np.pad(data, ((0, 0), (0, 48000 - data.shape[1])))

            results = sess.run(output_names, {input_name: data})

            for name, result in zip(output_names, results):
                if name in input_names and isinstance(result, np.ndarray):
                    if result.ndim == 3:
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
    return channel_max


def apply_smoothquant(onnx_path, channel_max, alpha=0.5):
    """
    SmoothQuant 적용 (v2, 버그 수정).

    공유 입력 그룹별로:
    1. 모든 downstream weight의 per-channel max를 합산하여 combined s 계산
    2. LayerNorm의 γ, β를 /s
    3. 각 weight를 *s
    """
    print(f"\n{'='*70}")
    print(f"SmoothQuant v2 (α={alpha})")
    print(f"{'='*70}")

    model = onnx.load(onnx_path)
    groups = find_linear_groups(model)

    # Initializer를 dict로
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    smoothed_groups = 0
    smoothed_linears = 0
    skipped_no_ln = 0
    skipped_no_stats = 0

    # 이미 수정한 LayerNorm 추적 (같은 LN에 두 번 적용 방지)
    modified_layernorms = set()

    for input_name, members in groups.items():
        # activation stats 확인
        if input_name not in channel_max:
            skipped_no_stats += 1
            continue

        act_max = channel_max[input_name]
        n_channels = len(act_max)

        # LayerNorm 파라미터 찾기
        ln_params = find_layernorm_params(model, input_name)
        if ln_params is None:
            skipped_no_ln += 1
            continue

        gamma_name, beta_name = ln_params

        # 이미 이 LN을 수정했는지 확인 (정상적으론 한 번만 수정)
        ln_key = (gamma_name, beta_name)
        if ln_key in modified_layernorms:
            print(f"  WARNING: LayerNorm {gamma_name} already modified, skipping")
            continue
        modified_layernorms.add(ln_key)

        gamma = init_map.get(gamma_name)
        beta = init_map.get(beta_name)
        if gamma is None or beta is None:
            skipped_no_ln += 1
            continue

        if len(gamma) != n_channels or len(beta) != n_channels:
            print(f"  WARNING: γ/β size mismatch ({len(gamma)}) != act channels ({n_channels})")
            continue

        # Combined weight max: 모든 downstream weight의 per-channel max 중 최대값
        combined_weight_max = np.zeros(n_channels)
        valid_weights = []

        for member in members:
            weight = init_map.get(member['weight'])
            if weight is None:
                continue

            # weight shape 결정
            if weight.shape[0] == n_channels:
                # [in, out] (transposed) — per-channel max over output dim
                w_max = np.max(np.abs(weight), axis=1)
            elif weight.shape[1] == n_channels:
                # [out, in] — per-channel max over output dim
                w_max = np.max(np.abs(weight), axis=0)
            else:
                continue

            combined_weight_max = np.maximum(combined_weight_max, w_max)
            valid_weights.append(member)

        if not valid_weights:
            continue

        # Smoothing factor 계산
        act_max_safe = np.clip(act_max, 1e-5, None)
        w_max_safe = np.clip(combined_weight_max, 1e-5, None)

        s = np.power(act_max_safe, alpha) / np.power(w_max_safe, 1 - alpha)
        s = np.clip(s, 1e-5, 1e5)

        # 1. LayerNorm γ, β를 /s
        new_gamma = gamma / s
        new_beta = beta / s

        for init in model.graph.initializer:
            if init.name == gamma_name:
                init.CopyFrom(numpy_helper.from_array(new_gamma.astype(np.float32), gamma_name))
            elif init.name == beta_name:
                init.CopyFrom(numpy_helper.from_array(new_beta.astype(np.float32), beta_name))
        init_map[gamma_name] = new_gamma
        init_map[beta_name] = new_beta

        # 2. 각 downstream weight를 *s (입력 채널 방향)
        for member in valid_weights:
            weight = init_map[member['weight']]

            if weight.shape[0] == n_channels:
                # [in, out]
                new_weight = weight * s[:, np.newaxis]
            else:
                # [out, in]
                new_weight = weight * s[np.newaxis, :]

            for init in model.graph.initializer:
                if init.name == member['weight']:
                    init.CopyFrom(numpy_helper.from_array(
                        new_weight.astype(np.float32), member['weight']))
            init_map[member['weight']] = new_weight

            smoothed_linears += 1

        smoothed_groups += 1

        # 로그
        act_range_before = act_max.max()
        act_range_after = (act_max / s).max()
        w_range_before = combined_weight_max.max()
        w_range_after = (combined_weight_max * s).max()
        print(f"  Group '{input_name[-55:]}' ({len(valid_weights)} linears): "
              f"act {act_range_before:.2f}→{act_range_after:.2f}, "
              f"weight {w_range_before:.4f}→{w_range_after:.4f}")

    print(f"\n  Summary: {smoothed_groups} groups ({smoothed_linears} linears) smoothed")
    print(f"  Skipped: {skipped_no_ln} no LayerNorm, {skipped_no_stats} no activation stats")

    return model


def evaluate(path_or_model, label, test_input):
    """FP32 평가"""
    if isinstance(path_or_model, str):
        path = path_or_model
        cleanup = False
    else:
        f = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        path = f.name
        f.close()
        onnx.save(path_or_model, path)
        cleanup = True

    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: test_input})[0]
    finally:
        if cleanup:
            os.unlink(path)

    logit_range = output.max() - output.min()
    top2 = np.sort(output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]

    with open(os.path.join(BASE_DIR, "vocab.json")) as f:
        vd = json.load(f)
    vocab = [""] * (max(vd.values()) + 1)
    for c, i in vd.items():
        vocab[i] = c

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


def compare_outputs(orig, smoothed, label):
    """원본과 smoothed 출력 비교"""
    diff = np.abs(orig - smoothed)
    tokens_orig = np.argmax(orig[0], axis=-1)
    tokens_smooth = np.argmax(smoothed[0], axis=-1)
    agree = np.mean(tokens_orig == tokens_smooth) * 100

    print(f"  [{label}] FP32 일치도")
    print(f"    Max abs diff: {diff.max():.6f}")
    print(f"    Mean abs diff: {diff.mean():.6f}")
    print(f"    Argmax agreement: {agree:.1f}%")

    if agree < 99.0:
        # 불일치 위치 확인
        mismatch = np.where(tokens_orig != tokens_smooth)[0]
        print(f"    Mismatch positions ({len(mismatch)}): {mismatch[:10]}...")


def simulate_uint8_quant(path_or_model, test_input, label):
    """
    uint8 per-tensor 양자화 시뮬레이션.
    Weight만 양자화 (activation은 Pegasus가 처리).
    """
    if isinstance(path_or_model, str):
        model = onnx.load(path_or_model)
    else:
        model = copy.deepcopy(path_or_model)

    for init in model.graph.initializer:
        data = numpy_helper.to_array(init)
        if data.size < 100:
            continue
        fmin, fmax = float(data.min()), float(data.max())
        if fmax - fmin < 1e-10:
            continue
        scale = (fmax - fmin) / 255.0
        zp = int(np.round(-fmin / scale))
        zp = np.clip(zp, 0, 255)
        q = np.clip(np.round(data / scale + zp), 0, 255).astype(np.uint8)
        dq = (q.astype(np.float32) - zp) * scale
        init.CopyFrom(numpy_helper.from_array(dq, init.name))

    return evaluate(model, f"{label} (weight uint8 sim)", test_input)


def main():
    print("=" * 70)
    print("SmoothQuant v2 — Correct Implementation")
    print("=" * 70)

    test_input = np.load(os.path.join(BASE_DIR, "test_audio.npy")).astype(np.float32)
    print(f"Test input: {test_input.shape}")

    # 원본 평가
    print("\n--- Original Model ---")
    orig_output = evaluate(KO_ONNX, "Original FP32", test_input)

    # Activation 통계 수집
    channel_max = collect_activation_stats(KO_ONNX, CALIB_DIR, num_samples=50)

    # 여러 alpha 값
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        model = apply_smoothquant(KO_ONNX, channel_max, alpha=alpha)

        out_path = os.path.join(BASE_DIR,
                                f"wav2vec2_ko_base_3s_sqv2_a{alpha}_nopad10_opset12_sim.onnx")
        onnx.save(model, out_path)
        print(f"  Saved: {os.path.basename(out_path)} ({os.path.getsize(out_path)/1e6:.1f}MB)")

        # FP32 평가 — 이것이 원본과 동일해야 함
        smooth_output = evaluate(out_path, f"SQv2 α={alpha} FP32", test_input)

        # FP32 일치도 비교
        compare_outputs(orig_output, smooth_output, f"α={alpha}")

        # Weight uint8 시뮬레이션
        simulate_uint8_quant(out_path, test_input, f"SQv2 α={alpha}")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()

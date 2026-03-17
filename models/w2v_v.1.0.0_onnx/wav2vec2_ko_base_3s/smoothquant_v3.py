#!/usr/bin/env python3
"""
SmoothQuant v3 for Korean wav2vec2.

v2 실패 원인: wav2vec2의 잔차 연결이 LayerNorm 출력을 직접 사용.
  x_new = LN(x) + Attention(LN(x))
  γ/β를 /s하면 잔차도 /s → FP32 파괴.

v3 해결: γ/β를 수정하지 않고, MatMul 입력 전에 Div(s) 노드 삽입.
  LN_output → Div(s) → MatMul(s*W) = LN_output @ W (수학적 등가)
             → Residual Add (변경 없음)

Div(s) 출력의 activation 범위가 좁아져서 uint8 양자화 정밀도 개선.
"""
import os
import glob
import copy
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import onnxruntime as ort
import json
import tempfile
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KO_ONNX = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx")
CALIB_DIR = os.path.join(BASE_DIR, "aug_calib_npy")


def find_linear_groups(model):
    """Linear 레이어를 입력 텐서로 그룹화"""
    init_names = {init.name for init in model.graph.initializer}
    init_shapes = {}
    for init in model.graph.initializer:
        init_shapes[init.name] = list(init.dims)

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

        groups[input_name].append({
            'matmul_node': node,
            'matmul_name': node.name,
            'weight': weight_name,
            'weight_shape': init_shapes[weight_name],
        })

    return groups


def is_layernorm_output(model, tensor_name):
    """tensor_name이 LayerNorm의 출력(Mul→Add 패턴)인지 확인"""
    init_names = {init.name for init in model.graph.initializer}
    output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    if tensor_name not in output_to_node:
        return False

    node = output_to_node[tensor_name]
    if node.op_type != "Add":
        return False

    # Add의 한 입력이 initializer (β)
    has_beta = any(inp in init_names for inp in node.input)
    if not has_beta:
        return False

    # 나머지 입력이 Mul 출력 (γ * normalized)
    for inp in node.input:
        if inp not in init_names and inp in output_to_node:
            mul = output_to_node[inp]
            if mul.op_type == "Mul":
                has_gamma = any(i in init_names for i in mul.input)
                if has_gamma:
                    return True
    return False


def collect_activation_stats(onnx_path, calib_dir, input_names, num_samples=50):
    """per-channel activation max abs 수집"""
    print(f"\nCollecting activation statistics from {num_samples} samples...")

    model = onnx.load(onnx_path)
    existing_outputs = {o.name for o in model.graph.output}
    shape_info = {vi.name: vi for vi in model.graph.value_info}

    for name in input_names:
        if name not in existing_outputs:
            if name in shape_info:
                model.graph.output.append(shape_info[name])
            else:
                model.graph.output.append(
                    helper.make_tensor_value_info(name, TensorProto.FLOAT, None))

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name
        onnx.save(model, tmp_path)

    try:
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        channel_max = {}
        calib_files = sorted(glob.glob(os.path.join(calib_dir, "*.npy")))[:num_samples]
        print(f"  Using {len(calib_files)} calibration files")

        for i, cf in enumerate(calib_files):
            data = np.load(cf).astype(np.float32)
            if data.shape != (1, 48000):
                data = data.reshape(1, -1)[:, :48000]
                if data.shape[1] < 48000:
                    data = np.pad(data, ((0, 0), (0, 48000 - data.shape[1])))

            results = sess.run(out_names, {in_name: data})
            for name, result in zip(out_names, results):
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

    return channel_max


def apply_smoothquant_v3(onnx_path, channel_max, alpha=0.5):
    """
    SmoothQuant v3: Div(s) 노드 삽입 방식.

    각 LayerNorm output → Linear 그룹에 대해:
    1. Div(s) 노드 삽입 (LN_out → LN_out_smooth)
    2. MatMul 입력을 LN_out_smooth로 변경
    3. Weight를 *s
    4. 잔차 연결은 원래 LN_out 사용 (변경 없음)
    """
    print(f"\n{'='*70}")
    print(f"SmoothQuant v3 (α={alpha}) — Div node insertion")
    print(f"{'='*70}")

    model = onnx.load(onnx_path)
    groups = find_linear_groups(model)

    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    smoothed_groups = 0
    inserted_nodes = 0

    for input_name, members in groups.items():
        if input_name not in channel_max:
            continue

        # LayerNorm 출력인지 확인
        if not is_layernorm_output(model, input_name):
            continue

        act_max = channel_max[input_name]
        n_channels = len(act_max)

        # Combined weight max
        combined_w_max = np.zeros(n_channels)
        valid_members = []
        for member in members:
            w = init_map.get(member['weight'])
            if w is None or w.shape[0] != n_channels:
                continue
            w_max = np.max(np.abs(w), axis=1)  # [in_dim]
            combined_w_max = np.maximum(combined_w_max, w_max)
            valid_members.append(member)

        if not valid_members:
            continue

        # Compute s
        act_safe = np.clip(act_max, 1e-5, None)
        w_safe = np.clip(combined_w_max, 1e-5, None)
        s = np.power(act_safe, alpha) / np.power(w_safe, 1 - alpha)
        s = np.clip(s, 1e-5, 1e5)

        # 1. Div(s) 노드 삽입
        smooth_output_name = input_name + "_smooth"
        s_const_name = f"smooth_s_{smoothed_groups}"

        # s를 initializer로 추가
        s_tensor = numpy_helper.from_array(s.astype(np.float32), s_const_name)
        model.graph.initializer.append(s_tensor)

        # Div 노드 생성
        div_node = helper.make_node(
            "Div",
            inputs=[input_name, s_const_name],
            outputs=[smooth_output_name],
            name=f"SmoothQuant_Div_{smoothed_groups}",
        )

        # Div 노드를 그래프에 삽입 (첫 번째 consumer MatMul 앞에)
        first_matmul_idx = None
        for i, node in enumerate(model.graph.node):
            if any(node is m['matmul_node'] for m in valid_members):
                first_matmul_idx = i
                break

        if first_matmul_idx is not None:
            model.graph.node.insert(first_matmul_idx, div_node)
            inserted_nodes += 1

        # 2. MatMul 입력을 smooth_output_name으로 변경
        for member in valid_members:
            matmul = member['matmul_node']
            for j, inp in enumerate(matmul.input):
                if inp == input_name:
                    matmul.input[j] = smooth_output_name

        # 3. Weight를 *s
        for member in valid_members:
            w = init_map[member['weight']]
            new_w = w * s[:, np.newaxis]
            for init in model.graph.initializer:
                if init.name == member['weight']:
                    init.CopyFrom(numpy_helper.from_array(
                        new_w.astype(np.float32), member['weight']))
            init_map[member['weight']] = new_w

        act_before = act_max.max()
        act_after = (act_max / s).max()
        w_before = combined_w_max.max()
        w_after = (combined_w_max * s).max()
        print(f"  Group '{input_name[-55:]}' ({len(valid_members)} linears): "
              f"act {act_before:.2f}→{act_after:.2f}, "
              f"weight {w_before:.4f}→{w_after:.4f}")

        smoothed_groups += 1

    print(f"\n  Summary: {smoothed_groups} groups smoothed, {inserted_nodes} Div nodes inserted")
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
        output = sess.run(None, {sess.get_inputs()[0].name: test_input})[0]
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
    diff = np.abs(orig - smoothed)
    tokens_orig = np.argmax(orig[0], axis=-1)
    tokens_smooth = np.argmax(smoothed[0], axis=-1)
    agree = np.mean(tokens_orig == tokens_smooth) * 100

    print(f"  [{label}] FP32 일치도")
    print(f"    Max abs diff: {diff.max():.6f}")
    print(f"    Mean abs diff: {diff.mean():.6f}")
    print(f"    Argmax agreement: {agree:.1f}%")


def simulate_uint8_quant(path_or_model, test_input, label):
    """uint8 per-tensor weight 양자화 시뮬레이션"""
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
    print("SmoothQuant v3 — Div Node Insertion (residual-safe)")
    print("=" * 70)

    test_input = np.load(os.path.join(BASE_DIR, "test_audio.npy")).astype(np.float32)
    print(f"Test input: {test_input.shape}")

    # Original
    print("\n--- Original Model ---")
    orig_output = evaluate(KO_ONNX, "Original FP32", test_input)

    # Find smoothable groups
    model_tmp = onnx.load(KO_ONNX)
    groups = find_linear_groups(model_tmp)
    smoothable = [name for name in groups.keys() if is_layernorm_output(model_tmp, name)]
    print(f"\nSmoothable groups (LayerNorm → Linear): {len(smoothable)}")

    # Collect stats
    channel_max = collect_activation_stats(KO_ONNX, CALIB_DIR, smoothable, num_samples=50)

    # Test alphas
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        model = apply_smoothquant_v3(KO_ONNX, channel_max, alpha=alpha)

        out_path = os.path.join(BASE_DIR,
                                f"wav2vec2_ko_base_3s_sqv3_a{alpha}_nopad10_opset12_sim.onnx")
        onnx.save(model, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved: {os.path.basename(out_path)} ({size_mb:.1f}MB)")

        # FP32 — should be identical to original
        smooth_output = evaluate(out_path, f"SQv3 α={alpha} FP32", test_input)
        compare_outputs(orig_output, smooth_output, f"α={alpha}")

        # Weight uint8 simulation
        simulate_uint8_quant(out_path, test_input, f"SQv3 α={alpha}")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()

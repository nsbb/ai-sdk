#!/usr/bin/env python3
"""
Q, K 벡터에 Clip 추가하여 Q@K^T 범위를 간접적으로 제한.

Q@K^T 직접 clip 대비 장점:
- attention score의 상대 순서 보존
- Q, K 벡터의 outlier만 제거
- 결과적으로 Q@K^T 범위도 줄어듦

Q@K^T_max ≤ d * clip_q * clip_k (d=head_dim=64, 12 heads)
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort
import json
import glob
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KO_ONNX = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx")
TEST_INPUT_NPY = os.path.join(BASE_DIR, "test_audio.npy")
CALIB_DIR = os.path.join(BASE_DIR, "aug_calib_npy")


def find_qk_projections(model):
    """
    Q, K projection의 output을 찾는다.
    패턴: MatMul(input, q/k_weight) → Add(_, q/k_bias) → Reshape → Transpose → MatMul(Q, K^T)
    """
    init_names = {init.name for init in model.graph.initializer}
    output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    # 먼저 attention Q@K^T MatMul을 찾기
    softmax_nodes = [n for n in model.graph.node if n.op_type == "Softmax"]

    results = []

    for sm_node in softmax_nodes:
        # Softmax 입력을 역추적하여 attention MatMul 찾기
        current = sm_node.input[0]
        attn_matmul = None
        for _ in range(5):
            if current in output_to_node:
                prev = output_to_node[current]
                if prev.op_type == "MatMul":
                    attn_matmul = prev
                    break
                non_init = [i for i in prev.input if i not in init_names and i in output_to_node]
                if non_init:
                    current = non_init[0]
                else:
                    break
            else:
                break

        if attn_matmul is None:
            continue

        # Q@K^T MatMul의 두 입력 역추적 — 각각 Q, K^T
        q_input = attn_matmul.input[0]  # Q (after Transpose)
        k_input = attn_matmul.input[1]  # K^T (after Transpose)

        # Transpose → Reshape → Add(bias) → MatMul(weight) 역추적
        def trace_back_to_add(tensor_name, depth=0):
            if depth > 6 or tensor_name not in output_to_node:
                return None
            node = output_to_node[tensor_name]
            if node.op_type == "Add":
                # bias를 사용하는 Add = projection output
                for inp in node.input:
                    if inp in init_names:
                        return node  # This Add has a bias initializer = projection Add
                # 둘 다 non-init이면 계속 추적
                for inp in node.input:
                    if inp in output_to_node:
                        result = trace_back_to_add(inp, depth + 1)
                        if result:
                            return result
            elif node.op_type in ("Transpose", "Reshape"):
                return trace_back_to_add(node.input[0], depth + 1)
            return None

        q_add = trace_back_to_add(q_input)
        k_add = trace_back_to_add(k_input)

        results.append({
            'attn_matmul': attn_matmul,
            'q_add': q_add,
            'k_add': k_add,
            'q_input': q_input,  # Q input to attention MatMul (after Reshape/Transpose)
            'k_input': k_input,  # K input to attention MatMul (after Reshape/Transpose)
        })

    return results


def add_clip_node(model, target_output, clip_min, clip_max, tag):
    """target_output 뒤에 Clip 노드 삽입"""
    old_name = target_output
    new_name = old_name + f"_pre{tag}"

    # 원래 출력 이름을 가진 노드 찾기
    target_node = None
    target_idx = None
    for i, node in enumerate(model.graph.node):
        if old_name in node.output:
            target_node = node
            target_idx = i
            break

    if target_node is None:
        return False

    # 출력 이름 변경
    for j, out in enumerate(target_node.output):
        if out == old_name:
            target_node.output[j] = new_name

    # Clip 파라미터
    min_name = f"{tag}_min_{old_name}"
    max_name = f"{tag}_max_{old_name}"
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(clip_min, dtype=np.float32), min_name))
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(clip_max, dtype=np.float32), max_name))

    clip_node = onnx.helper.make_node(
        "Clip", [new_name, min_name, max_name], [old_name],
        name=f"Clip_{tag}_{old_name}")

    model.graph.node.insert(target_idx + 1, clip_node)
    return True


def evaluate(model_path_or_bytes, test_input, label):
    """평가"""
    if isinstance(model_path_or_bytes, str):
        sess = ort.InferenceSession(model_path_or_bytes, providers=["CPUExecutionProvider"])
    else:
        sess = ort.InferenceSession(model_path_or_bytes, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: test_input})[0]

    logit_range = output.max() - output.min()
    top2 = np.sort(output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]

    with open(os.path.join(BASE_DIR, "vocab.json")) as f:
        vocab_dict = json.load(f)
    vocab = [""] * (max(vocab_dict.values()) + 1)
    for char, idx in vocab_dict.items():
        vocab[idx] = char

    tokens = np.argmax(output[0], axis=-1)
    prev = -1; decoded = []
    for t in tokens:
        if t != prev:
            if t != 0: decoded.append(t)
            prev = t
    text = "".join(vocab[t] for t in decoded if t < len(vocab))

    print(f"  [{label}]")
    print(f"    Logit range: {logit_range:.2f}, Top1-2 gap: {gaps.mean():.3f}")
    print(f"    Decoded: {text}")
    return output


def measure_qk_ranges(model_path, calib_dir, n_samples=20):
    """calibration 데이터로 Q, K 벡터의 실제 범위 측정"""
    model = onnx.load(model_path)
    projections = find_qk_projections(model)

    # Q, K projection output을 그래프 출력에 추가
    existing_outputs = {o.name for o in model.graph.output}
    shape_info = {vi.name: vi for vi in model.graph.value_info}

    output_names_to_track = []
    for proj in projections:
        for key in ['q_input', 'k_input']:
            name = proj[key]
            if name not in existing_outputs:
                if name in shape_info:
                    model.graph.output.append(shape_info[name])
                else:
                    model.graph.output.append(
                        onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
                output_names_to_track.append(name)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp = f.name
        onnx.save(model, tmp)

    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    all_output_names = [o.name for o in sess.get_outputs()]

    q_ranges = []
    k_ranges = []

    calib_files = sorted(glob.glob(os.path.join(calib_dir, "*.npy")))[:n_samples]
    for cf in calib_files:
        data = np.load(cf).astype(np.float32)
        if data.shape != (1, 48000):
            data = data.reshape(1, -1)[:, :48000]
            if data.shape[1] < 48000:
                data = np.pad(data, ((0, 0), (0, 48000 - data.shape[1])))

        results = sess.run(all_output_names, {input_name: data})
        result_map = dict(zip(all_output_names, results))

        for proj in projections:
            if proj['q_input'] in result_map:
                q = result_map[proj['q_input']]
                q_ranges.append(float(np.max(np.abs(q))))
            if proj['k_input'] in result_map:
                k = result_map[proj['k_input']]
                k_ranges.append(float(np.max(np.abs(k))))

    os.unlink(tmp)

    q_ranges = np.array(q_ranges).reshape(-1, len(projections))
    k_ranges = np.array(k_ranges).reshape(-1, len(projections))

    print(f"\nQ vector max abs per layer (mean over {n_samples} samples):")
    for i in range(len(projections)):
        print(f"  Layer {i:2d}: Q={q_ranges[:, i].mean():.3f}  K={k_ranges[:, i].mean():.3f}  "
              f"Q*K*64 est: {q_ranges[:, i].mean() * k_ranges[:, i].mean() * 64:.1f}")

    return q_ranges, k_ranges


def main():
    test_input = np.load(TEST_INPUT_NPY)

    print("=" * 70)
    print("Q/K Vector Clamping for Korean Wav2Vec2")
    print("=" * 70)

    # 원본 평가
    evaluate(KO_ONNX, test_input, "Original FP32")

    # Q, K 범위 측정
    print("\n--- Measuring Q, K vector ranges ---")
    q_ranges, k_ranges = measure_qk_ranges(KO_ONNX, CALIB_DIR, n_samples=20)

    # Q, K projection 위치 찾기
    model = onnx.load(KO_ONNX)
    projections = find_qk_projections(model)
    print(f"\nFound {len(projections)} attention layers")

    # 실험: Q, K 벡터를 Transpose 후 (attention MatMul 입력) clamp
    for clip_val in [3.0, 2.0, 1.5, 1.0]:
        print(f"\n{'='*70}")
        print(f"Q/K vector clip = [-{clip_val}, {clip_val}]")
        print(f"  Theoretical Q@K^T max = 64 * {clip_val}^2 = {64 * clip_val**2:.0f}")

        model = onnx.load(KO_ONNX)
        projections = find_qk_projections(model)

        clipped = 0
        for i, proj in enumerate(projections):
            # Q input (before attention MatMul)
            if add_clip_node(model, proj['q_input'], -clip_val, clip_val, f"qclip_L{i}"):
                clipped += 1
            # K input (before attention MatMul)
            if add_clip_node(model, proj['k_input'], -clip_val, clip_val, f"kclip_L{i}"):
                clipped += 1

        print(f"  Inserted {clipped} Clip nodes")

        out_path = os.path.join(BASE_DIR,
            f"wav2vec2_ko_base_3s_qkclip{clip_val}_nopad10_opset12_sim.onnx")
        onnx.save(model, out_path)
        print(f"  Saved: {os.path.basename(out_path)}")

        evaluate(out_path, test_input, f"QK clip={clip_val}")

    # === 최적 조합: Q@K^T clip=50 + Q/K clip=3.0 ===
    print(f"\n{'='*70}")
    print("Combined: Q/K clip=3.0 + Q@K^T clip=50")

    model = onnx.load(KO_ONNX)
    projections = find_qk_projections(model)

    for i, proj in enumerate(projections):
        add_clip_node(model, proj['q_input'], -3.0, 3.0, f"qclip2_L{i}")
        add_clip_node(model, proj['k_input'], -3.0, 3.0, f"kclip2_L{i}")

    # attention MatMul (Q@K^T) 뒤에도 clip
    from add_activation_clamp import find_attention_matmul_nodes, add_clip_after_node
    attn_matmuls = find_attention_matmul_nodes(model)
    for i, node in enumerate(attn_matmuls):
        add_clip_after_node(model, node, -50, 50, suffix=f"_combo{i}")

    combo_path = os.path.join(BASE_DIR,
        "wav2vec2_ko_base_3s_qkclip3_attnclip50_nopad10_opset12_sim.onnx")
    onnx.save(model, combo_path)
    print(f"  Saved: {os.path.basename(combo_path)}")
    evaluate(combo_path, test_input, "QK=3 + Attn=50")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Korean wav2vec2 양자화 오류 레이어별 분석.

각 레이어의 weight 동적 범위 + 중간 activation 동적 범위를 분석하여
uint8 양자화에서 어디서 오류가 누적되는지 식별.

또한 weight clipping + channel-wise 범위 분석 수행.
"""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KO_ONNX = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx")
EN_ONNX = os.path.join(os.path.dirname(BASE_DIR),
                        "wav2vec2_base_960h_5s_uint8",
                        "wav2vec2_base_960h_5s.onnx")
TEST_INPUT_NPY = os.path.join(BASE_DIR, "test_audio.npy")


def simulate_uint8_quant(tensor):
    """uint8 asymmetric_affine 양자화 시뮬레이션"""
    fmin, fmax = tensor.min(), tensor.max()
    if fmax == fmin:
        return tensor, 1.0, 0
    scale = (fmax - fmin) / 255.0
    zp = int(np.round(-fmin / scale))
    zp = np.clip(zp, 0, 255)
    quantized = np.clip(np.round(tensor / scale + zp), 0, 255).astype(np.uint8)
    dequantized = (quantized.astype(np.float32) - zp) * scale
    return dequantized, scale, zp


def analyze_weight_ranges(model, label):
    """모든 weight의 동적 범위 분석"""
    print(f"\n{'='*70}")
    print(f"Weight Analysis: {label}")
    print(f"{'='*70}")

    problematic = []
    all_stats = []

    for init in model.graph.initializer:
        data = numpy_helper.to_array(init)
        if data.size < 100:  # 작은 텐서 무시
            continue

        fmin, fmax = data.min(), data.max()
        drange = fmax - fmin
        scale = drange / 255.0

        stats = {
            'name': init.name,
            'shape': list(init.dims),
            'range': drange,
            'scale': scale,
            'min': fmin,
            'max': fmax,
            'mean_abs': np.mean(np.abs(data)),
            'std': np.std(data),
            'outlier_ratio': np.mean(np.abs(data) > 3 * np.std(data)),
        }
        all_stats.append(stats)

        if scale > 0.5:  # uint8 step > 0.5 float
            problematic.append(stats)

    # 가장 큰 scale 순으로 정렬
    all_stats.sort(key=lambda x: -x['scale'])

    print(f"\nTop 20 largest quantization scales:")
    print(f"{'Name':<60} {'Shape':<20} {'Range':>10} {'Scale':>10} {'Outlier%':>10}")
    print("-" * 115)
    for s in all_stats[:20]:
        name_short = s['name'][-55:] if len(s['name']) > 55 else s['name']
        print(f"  {name_short:<58} {str(s['shape']):<18} "
              f"{s['range']:>10.4f} {s['scale']:>10.6f} {s['outlier_ratio']*100:>9.2f}%")

    print(f"\nTotal initializers: {len(all_stats)}")
    print(f"With scale > 0.5: {len(problematic)}")
    print(f"With scale > 1.0: {len([s for s in all_stats if s['scale'] > 1.0])}")

    return all_stats


def analyze_activations(onnx_path, input_data, label):
    """중간 activation 동적 범위 분석 (ONNX Runtime intermediate outputs)"""
    print(f"\n{'='*70}")
    print(f"Activation Analysis: {label}")
    print(f"{'='*70}")

    model = onnx.load(onnx_path)

    # 모든 중간 출력을 추적하도록 모델 수정
    # 각 노드의 output을 graph output에 추가
    intermediate_names = []
    for node in model.graph.node:
        for output in node.output:
            intermediate_names.append(output)

    # 너무 많으면 attention 관련 노드만
    # MatMul, Softmax, Add 노드의 출력만 추적
    attention_outputs = []
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Softmax", "Add", "Mul", "Div"):
            for output in node.output:
                attention_outputs.append((output, node.op_type, node.name))

    # graph output에 추가
    existing_outputs = {o.name for o in model.graph.output}
    shape_info = {vi.name: vi for vi in model.graph.value_info}

    for out_name, op_type, node_name in attention_outputs:
        if out_name not in existing_outputs:
            if out_name in shape_info:
                model.graph.output.append(shape_info[out_name])
            else:
                # shape 정보 없으면 빈 텐서 타입으로
                from onnx import TensorProto
                model.graph.output.append(
                    onnx.helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
                )

    # 수정된 모델로 추론
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name
        onnx.save(model, tmp_path)

    try:
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: input_data})
        output_names = [o.name for o in sess.get_outputs()]
    finally:
        os.unlink(tmp_path)

    # activation 통계
    act_stats = []
    for name, data in zip(output_names, outputs):
        if isinstance(data, np.ndarray) and data.size > 10:
            fmin, fmax = data.min(), data.max()
            drange = fmax - fmin
            scale = drange / 255.0 if drange > 0 else 0

            # uint8 양자화 시뮬레이션
            dequant, _, _ = simulate_uint8_quant(data)
            quant_error = np.mean(np.abs(data - dequant))
            relative_error = quant_error / (np.mean(np.abs(data)) + 1e-10)

            # 해당 op_type 찾기
            op_type = "?"
            for out_name, ot, nn in attention_outputs:
                if out_name == name:
                    op_type = ot
                    break

            act_stats.append({
                'name': name,
                'op_type': op_type,
                'shape': list(data.shape),
                'range': drange,
                'scale': scale,
                'quant_error': quant_error,
                'relative_error': relative_error,
                'min': fmin,
                'max': fmax,
            })

    # scale 큰 순 정렬
    act_stats.sort(key=lambda x: -x['scale'])

    print(f"\nTop 30 activations with largest uint8 scales:")
    print(f"{'Op':<8} {'Name':<50} {'Range':>10} {'Scale':>8} {'RelErr':>8}")
    print("-" * 90)
    for s in act_stats[:30]:
        name_short = s['name'][-46:] if len(s['name']) > 46 else s['name']
        print(f"  {s['op_type']:<6} {name_short:<48} "
              f"{s['range']:>10.2f} {s['scale']:>8.4f} {s['relative_error']:>8.4f}")

    # Softmax 출력은 [0,1] 범위이므로 양자화 잘 되어야 함
    softmax_stats = [s for s in act_stats if s['op_type'] == 'Softmax']
    matmul_stats = [s for s in act_stats if s['op_type'] == 'MatMul']

    print(f"\nSoftmax outputs: {len(softmax_stats)} nodes")
    print(f"  avg range: {np.mean([s['range'] for s in softmax_stats]):.4f}")
    print(f"  avg scale: {np.mean([s['scale'] for s in softmax_stats]):.6f}")

    print(f"\nMatMul outputs: {len(matmul_stats)} nodes")
    if matmul_stats:
        print(f"  avg range: {np.mean([s['range'] for s in matmul_stats]):.2f}")
        print(f"  avg scale: {np.mean([s['scale'] for s in matmul_stats]):.4f}")
        print(f"  max range: {max([s['range'] for s in matmul_stats]):.2f}")

    return act_stats


def compare_en_ko_activations(ko_stats, en_stats):
    """영어 vs 한국어 activation 범위 비교"""
    print(f"\n{'='*70}")
    print("English vs Korean Activation Range Comparison")
    print(f"{'='*70}")

    # MatMul만 비교 (가장 중요)
    ko_mm = [s for s in ko_stats if s['op_type'] == 'MatMul']
    en_mm = [s for s in en_stats if s['op_type'] == 'MatMul']

    print(f"\nMatMul activation ranges:")
    print(f"  Korean: {len(ko_mm)} nodes, avg_range={np.mean([s['range'] for s in ko_mm]):.2f}, "
          f"max_range={max([s['range'] for s in ko_mm]):.2f}")
    print(f"  English: {len(en_mm)} nodes, avg_range={np.mean([s['range'] for s in en_mm]):.2f}, "
          f"max_range={max([s['range'] for s in en_mm]):.2f}")

    # 한국어에서 가장 문제가 되는 MatMul (range > 100)
    ko_problem = [s for s in ko_mm if s['range'] > 100]
    print(f"\n  Korean MatMul with range > 100: {len(ko_problem)}")
    for s in sorted(ko_problem, key=lambda x: -x['range'])[:10]:
        name_short = s['name'][-50:] if len(s['name']) > 50 else s['name']
        print(f"    {name_short}: range={s['range']:.1f}, scale={s['scale']:.4f}")


def try_weight_clipping(model_path, percentile=99.9):
    """Weight clipping: outlier 제거하여 양자화 범위 축소"""
    print(f"\n{'='*70}")
    print(f"Weight Clipping Experiment (percentile={percentile})")
    print(f"{'='*70}")

    model = onnx.load(model_path)
    clipped_count = 0
    total_clipped_values = 0
    total_values = 0

    for init in model.graph.initializer:
        data = numpy_helper.to_array(init).copy()
        if data.size < 100:
            continue

        total_values += data.size
        lo = np.percentile(data, 100 - percentile)
        hi = np.percentile(data, percentile)

        n_clipped = np.sum((data < lo) | (data > hi))
        if n_clipped > 0:
            data_clipped = np.clip(data, lo, hi)
            new_tensor = numpy_helper.from_array(data_clipped.astype(np.float32), init.name)
            init.CopyFrom(new_tensor)
            clipped_count += 1
            total_clipped_values += n_clipped

    print(f"  Clipped {clipped_count} tensors, {total_clipped_values}/{total_values} values "
          f"({total_clipped_values/total_values*100:.4f}%)")

    out_path = model_path.replace(".onnx", f"_clip{percentile}.onnx")
    onnx.save(model, out_path)
    print(f"  Saved: {out_path}")

    # FP32 추론
    test_input = np.load(TEST_INPUT_NPY)
    output = run_fp32_inference(out_path, test_input)
    print(f"  Logit range: [{output.min():.2f}, {output.max():.2f}]")

    top2 = np.sort(output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]
    print(f"  Top1-Top2 gap: mean={gaps.mean():.3f}")

    import json
    vocab_path = os.path.join(BASE_DIR, "vocab.json")
    with open(vocab_path) as f:
        vocab_dict = json.load(f)
    vocab = [""] * (max(vocab_dict.values()) + 1)
    for char, idx in vocab_dict.items():
        vocab[idx] = char
    text = ctc_greedy_decode(output[0], vocab)
    print(f"  Decoded: {text}")

    return out_path


def run_fp32_inference(onnx_path, input_data):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: input_data})[0]


def ctc_greedy_decode(logits, vocab):
    tokens = np.argmax(logits, axis=-1)
    if len(tokens.shape) > 1:
        tokens = tokens[0]
    prev = -1
    decoded = []
    for t in tokens:
        if t != prev:
            if t != 0:
                decoded.append(t)
            prev = t
    text = ""
    for t in decoded:
        if t < len(vocab):
            text += vocab[t]
    return text


def try_cross_layer_equalization(model_path):
    """
    Cross-Layer Equalization for FFN layers.
    FFN: Linear1(768→3072) → GELU → Linear2(3072→768)
    Equalize: scale Linear1 output channels, inverse-scale Linear2 input channels.
    """
    print(f"\n{'='*70}")
    print("Cross-Layer Equalization (FFN layers)")
    print(f"{'='*70}")

    model = onnx.load(model_path)
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    # FFN weight pair 찾기: fc1 (768→3072), fc2 (3072→768)
    fc1_weights = []
    fc2_weights = []
    for name, data in init_map.items():
        if 'feed_forward' in name or 'fc1' in name or 'fc2' in name:
            if 'weight' in name:
                if data.shape == (3072, 768):
                    fc1_weights.append(name)
                elif data.shape == (768, 3072):
                    fc2_weights.append(name)

    if not fc1_weights:
        # 이름 패턴이 다를 수 있음 — shape 기반으로 찾기
        for name, data in init_map.items():
            if len(data.shape) == 2:
                if data.shape == (3072, 768) and 'weight' in name.lower():
                    fc1_weights.append(name)
                elif data.shape == (768, 3072) and 'weight' in name.lower():
                    fc2_weights.append(name)

    print(f"  Found {len(fc1_weights)} fc1 weights, {len(fc2_weights)} fc2 weights")

    if len(fc1_weights) != len(fc2_weights):
        print("  Mismatch in fc1/fc2 counts, skipping CLE")
        return None

    # 정렬 (같은 레이어끼리 매칭)
    fc1_weights.sort()
    fc2_weights.sort()

    equalized = 0
    for fc1_name, fc2_name in zip(fc1_weights, fc2_weights):
        w1 = init_map[fc1_name]  # [3072, 768]
        w2 = init_map[fc2_name]  # [768, 3072]

        # channel-wise max abs
        r1 = np.max(np.abs(w1), axis=1)  # [3072]
        r2 = np.max(np.abs(w2), axis=0)  # [3072]

        # equalization scale
        s = np.sqrt(r1 / (r2 + 1e-8))
        s = np.clip(s, 0.01, 100.0)

        # w1_new = diag(1/s) @ w1, w2_new = w2 @ diag(s)
        w1_new = w1 / s[:, np.newaxis]
        w2_new = w2 * s[np.newaxis, :]

        # 업데이트
        for init in model.graph.initializer:
            if init.name == fc1_name:
                new_t = numpy_helper.from_array(w1_new.astype(np.float32), fc1_name)
                init.CopyFrom(new_t)
            elif init.name == fc2_name:
                new_t = numpy_helper.from_array(w2_new.astype(np.float32), fc2_name)
                init.CopyFrom(new_t)

        equalized += 1

    print(f"  Equalized {equalized} FFN layer pairs")

    out_path = model_path.replace(".onnx", "_cle.onnx")
    onnx.save(model, out_path)
    print(f"  Saved: {out_path}")

    # FP32 추론
    test_input = np.load(TEST_INPUT_NPY)
    output = run_fp32_inference(out_path, test_input)
    print(f"  Logit range: [{output.min():.2f}, {output.max():.2f}]")

    top2 = np.sort(output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]
    print(f"  Top1-Top2 gap: mean={gaps.mean():.3f}")

    import json
    with open(os.path.join(BASE_DIR, "vocab.json")) as f:
        vocab_dict = json.load(f)
    vocab = [""] * (max(vocab_dict.values()) + 1)
    for char, idx in vocab_dict.items():
        vocab[idx] = char
    text = ctc_greedy_decode(output[0], vocab)
    print(f"  Decoded: {text}")

    return out_path


def main():
    test_input = np.load(TEST_INPUT_NPY)
    print(f"Test input: {test_input.shape}, range=[{test_input.min():.4f}, {test_input.max():.4f}]")

    # === Part 1: Weight 범위 분석 ===
    ko_model = onnx.load(KO_ONNX)
    en_model = onnx.load(EN_ONNX)

    ko_w_stats = analyze_weight_ranges(ko_model, "Korean")
    en_w_stats = analyze_weight_ranges(en_model, "English")

    # scale > 1.0인 텐서 비교
    ko_high = [s for s in ko_w_stats if s['scale'] > 1.0]
    en_high = [s for s in en_w_stats if s['scale'] > 1.0]
    print(f"\n--- Tensors with uint8 scale > 1.0 ---")
    print(f"  Korean: {len(ko_high)}")
    for s in ko_high:
        print(f"    {s['name'][-50:]}: scale={s['scale']:.4f}, range={s['range']:.2f}")
    print(f"  English: {len(en_high)}")
    for s in en_high:
        print(f"    {s['name'][-50:]}: scale={s['scale']:.4f}, range={s['range']:.2f}")

    # === Part 2: Activation 범위 분석 ===
    print("\n\n--- Activation Analysis (may take a while) ---")
    ko_act_stats = analyze_activations(KO_ONNX, test_input, "Korean")

    # 영어 모델은 입력 shape이 다름 (80000 vs 48000)
    en_input = np.random.randn(1, 80000).astype(np.float32) * 0.1
    en_act_stats = analyze_activations(EN_ONNX, en_input, "English")

    compare_en_ko_activations(ko_act_stats, en_act_stats)

    # === Part 3: Weight Clipping 실험 ===
    print("\n\n--- Weight Clipping Experiments ---")
    for pct in [99.99, 99.9, 99.5]:
        try_weight_clipping(KO_ONNX, percentile=pct)

    # === Part 4: Cross-Layer Equalization ===
    try_cross_layer_equalization(KO_ONNX)


if __name__ == "__main__":
    main()

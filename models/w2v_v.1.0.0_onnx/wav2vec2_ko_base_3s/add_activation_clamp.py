#!/usr/bin/env python3
"""
Korean wav2vec2에 activation clamping 추가.

문제: 한국어 모델의 중간 activation range가 영어 대비 2-5배 넓음
  - attention Q@K^T: 138-277 (영어: 대부분 < 50)
  - FFN output: 136-254 (영어: 30-69)
  → uint8 scale이 너무 커서 양자화 정밀도 부족

해결: 문제가 되는 activation 뒤에 Clip 노드 삽입
  → 동적 범위 제한 → uint8 scale 감소 → 양자화 정밀도 향상
  → FP32에서 약간의 정보 손실 (outlier clipping) 있지만,
     uint8 양자화 시 모든 값이 뭉개지는 것보다 나음

전략:
1. attention MatMul (Q@K^T) 뒤에 Clip(-C, C) 삽입
   - 이것은 softmax 전이므로 outlier 제거해도 attention 패턴 유지
2. FFN 중간 activation은 건드리지 않음 (GELU 영향)
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KO_ONNX = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx")
TEST_INPUT_NPY = os.path.join(BASE_DIR, "test_audio.npy")


def find_attention_matmul_nodes(model):
    """
    attention 패턴의 Q@K^T MatMul 노드를 찾는다.
    패턴: MatMul → Div(또는 Mul by 1/sqrt(d)) → Add(mask) → Softmax
    Q@K^T MatMul의 출력이 Softmax로 가는 경로에 있는 MatMul.
    """
    # Softmax 노드 수집
    softmax_nodes = [n for n in model.graph.node if n.op_type == "Softmax"]

    # 각 Softmax의 입력을 역추적하여 MatMul 찾기
    attn_matmuls = []

    # 출력→노드 매핑
    output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    for sm_node in softmax_nodes:
        # Softmax 입력을 역추적 (최대 3단계)
        current = sm_node.input[0]
        for _ in range(5):
            if current in output_to_node:
                prev_node = output_to_node[current]
                if prev_node.op_type == "MatMul":
                    attn_matmuls.append(prev_node)
                    break
                elif prev_node.op_type in ("Add", "Div", "Mul", "Where"):
                    # 다음 입력으로 역추적 (첫번째 non-initializer 입력)
                    init_names = {init.name for init in model.graph.initializer}
                    non_init_inputs = [inp for inp in prev_node.input
                                       if inp not in init_names and inp in output_to_node]
                    if non_init_inputs:
                        current = non_init_inputs[0]
                    else:
                        break
                else:
                    break
            else:
                break

    return attn_matmuls


def add_clip_after_node(model, target_node, clip_min, clip_max, suffix=""):
    """
    target_node의 출력 뒤에 Clip 노드를 삽입.
    """
    old_output = target_node.output[0]
    new_intermediate = old_output + f"_preclip{suffix}"

    # target_node의 출력 이름 변경
    target_node.output[0] = new_intermediate

    # Clip의 min/max를 initializer로 추가
    min_name = f"clip_min{suffix}_{old_output}"
    max_name = f"clip_max{suffix}_{old_output}"

    min_tensor = numpy_helper.from_array(
        np.array(clip_min, dtype=np.float32), min_name)
    max_tensor = numpy_helper.from_array(
        np.array(clip_max, dtype=np.float32), max_name)

    model.graph.initializer.append(min_tensor)
    model.graph.initializer.append(max_tensor)

    # Clip 노드 생성
    clip_node = onnx.helper.make_node(
        "Clip",
        inputs=[new_intermediate, min_name, max_name],
        outputs=[old_output],
        name=f"Clip_attn{suffix}_{old_output}"
    )

    # 노드 삽입 (target_node 바로 뒤)
    target_idx = None
    for i, node in enumerate(model.graph.node):
        if node is target_node:
            target_idx = i
            break

    if target_idx is not None:
        model.graph.node.insert(target_idx + 1, clip_node)

    return clip_node


def evaluate(onnx_path, test_input, label):
    """평가"""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
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
    print(f"    Logit range: {logit_range:.2f}, Top1-2 gap: {gaps.mean():.3f}, "
          f"min gap: {gaps.min():.3f}")
    print(f"    Decoded: {text}")
    return output


def main():
    print("=" * 70)
    print("Activation Clamping for Korean Wav2Vec2")
    print("=" * 70)

    test_input = np.load(TEST_INPUT_NPY)
    model = onnx.load(KO_ONNX)

    # attention Q@K^T MatMul 노드 찾기
    attn_matmuls = find_attention_matmul_nodes(model)
    print(f"\nFound {len(attn_matmuls)} attention MatMul (Q@K^T) nodes:")
    for node in attn_matmuls:
        print(f"  {node.name}: {node.output[0]}")

    # 원본 평가
    print(f"\n{'='*70}")
    evaluate(KO_ONNX, test_input, "Original FP32")

    # 여러 clamp 범위 시도
    for clip_val in [50, 30, 20, 10]:
        print(f"\n{'='*70}")
        print(f"Clip range: [-{clip_val}, {clip_val}]")
        print(f"  uint8 scale: {2*clip_val/255:.4f} (vs original ~1.08)")

        model = onnx.load(KO_ONNX)
        attn_matmuls = find_attention_matmul_nodes(model)

        for i, node in enumerate(attn_matmuls):
            add_clip_after_node(model, node, -clip_val, clip_val, suffix=f"_L{i}")

        out_path = os.path.join(BASE_DIR,
                                f"wav2vec2_ko_base_3s_clip{clip_val}_nopad10_opset12_sim.onnx")
        onnx.save(model, out_path)
        print(f"  Saved: {os.path.basename(out_path)}")

        evaluate(out_path, test_input, f"Clip={clip_val} FP32")

    # === 추가: FFN intermediate도 clamp ===
    print(f"\n{'='*70}")
    print("Full clamp: attention MatMul + all wide-range nodes")

    # activation 분석 결과에서 range > 100인 노드들:
    # - attention MatMul (이미 처리)
    # - FFN output dense MatMul
    # - CNN Div (feature extractor)
    # - residual Add

    # FFN output dense MatMul도 clip해보기
    model = onnx.load(KO_ONNX)
    attn_matmuls = find_attention_matmul_nodes(model)

    # attention MatMul clip=30
    for i, node in enumerate(attn_matmuls):
        add_clip_after_node(model, node, -30, 30, suffix=f"_attn{i}")

    # 모든 residual Add 뒤에도 clip 추가 (Add_1 = FFN output + residual)
    residual_adds = []
    for node in model.graph.node:
        if node.op_type == "Add":
            # "Add_1" 패턴 (residual connections)
            output_name = node.output[0]
            if "Add_1_output" in output_name:
                residual_adds.append(node)

    for i, node in enumerate(residual_adds):
        add_clip_after_node(model, node, -100, 100, suffix=f"_res{i}")

    out_path = os.path.join(BASE_DIR,
                            "wav2vec2_ko_base_3s_fullclip_nopad10_opset12_sim.onnx")
    onnx.save(model, out_path)
    print(f"  Saved: {os.path.basename(out_path)}")
    evaluate(out_path, test_input, "FullClip (attn=30, residual=100)")


if __name__ == "__main__":
    main()

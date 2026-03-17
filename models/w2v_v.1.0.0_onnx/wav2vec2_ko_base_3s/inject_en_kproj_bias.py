#!/usr/bin/env python3
"""
Korean wav2vec2에 영어 모델의 k_proj bias를 이식하여 attention 패턴 변경.

배경:
- 영어 모델: k_proj bias mean_abs=15.11 → sharp attention → Q@K^T range 9-17 → uint8 OK
- 한국어 모델: k_proj bias mean_abs=0.14 → flat attention → Q@K^T range 82-268 → uint8 파괴

전략:
1. 영어 k_proj bias 직접 이식 (attention anchor 생성)
2. 한국어 k_proj bias 스케일업 (기존 패턴 유지, 크기만 증폭)
3. FP32 추론으로 효과 확인
"""
import os
import numpy as np
import onnx
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KO_ONNX = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_nopad10_opset12_sim.onnx")
EN_ONNX = os.path.join(os.path.dirname(BASE_DIR),
                        "wav2vec2_base_960h_5s_uint8",
                        "wav2vec2_base_960h_5s.onnx")

# 테스트 입력
TEST_INPUT_NPY = os.path.join(BASE_DIR, "test_audio.npy")

def find_kproj_bias_names(model):
    """k_proj.bias 이름을 찾는다. ONNX graph에서 Add 노드의 bias 입력 중 k_proj 포함."""
    bias_names = []
    for node in model.graph.node:
        if node.op_type == "Add":
            for inp in node.input:
                if "k_proj" in inp and "bias" in inp.lower():
                    bias_names.append(inp)
    # 중복 제거, 순서 유지
    seen = set()
    result = []
    for name in bias_names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def find_kproj_bias_by_graph_trace(model):
    """
    ONNX 그래프를 추적하여 k_proj bias를 찾는다.
    attention 패턴: MatMul(input, q/k/v_proj.weight) → Add(_, q/k/v_proj.bias)
    k_proj는 보통 두번째 projection (q, k, v 순서).
    """
    # 먼저 모든 initializer 이름을 집합으로
    init_names = {init.name for init in model.graph.initializer}

    # Add 노드 중 하나의 입력이 initializer인 것 찾기
    add_with_bias = []
    for node in model.graph.node:
        if node.op_type == "Add":
            for inp in node.input:
                if inp in init_names:
                    # 이 initializer의 shape 확인
                    for init in model.graph.initializer:
                        if init.name == inp:
                            shape = list(init.dims)
                            if len(shape) == 1 and shape[0] == 768:
                                add_with_bias.append((node.name, inp, node))
                            break

    # attention layer 패턴: 같은 레이어의 q, k, v bias는 연속됨
    # 이름에서 layer index 추출 시도
    bias_info = []
    for node_name, bias_name, node in add_with_bias:
        bias_info.append(bias_name)

    return bias_info


def get_initializer(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return np.array(onnx.numpy_helper.to_array(init))
    return None


def set_initializer(model, name, new_data):
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            new_tensor = onnx.numpy_helper.from_array(new_data.astype(np.float32), name)
            model.graph.initializer[i].CopyFrom(new_tensor)
            return True
    return False


def analyze_biases(model, bias_names, label):
    """bias 통계 출력"""
    print(f"\n=== {label} k_proj biases ===")
    for i, name in enumerate(bias_names):
        data = get_initializer(model, name)
        if data is not None:
            print(f"  Layer {i:2d}: mean_abs={np.mean(np.abs(data)):8.4f}  "
                  f"max={np.max(data):8.4f}  min={np.min(data):8.4f}  "
                  f"std={np.std(data):8.4f}")


def identify_qkv_bias_groups(model):
    """
    attention 레이어별 q, k, v bias를 그룹으로 식별.
    wav2vec2 ONNX에서 각 layer의 attention은:
      MatMul → Add (q_proj)
      MatMul → Add (k_proj)
      MatMul → Add (v_proj)
    순서로 나타남.

    Returns: list of (q_bias_name, k_bias_name, v_bias_name) per layer
    """
    init_names = {init.name for init in model.graph.initializer}
    init_shapes = {}
    for init in model.graph.initializer:
        init_shapes[init.name] = list(init.dims)

    # 768-dim bias를 가진 Add 노드들을 순서대로 수집
    bias_768 = []
    for node in model.graph.node:
        if node.op_type == "Add":
            for inp in node.input:
                if inp in init_names and init_shapes.get(inp) == [768]:
                    bias_768.append(inp)

    # MatMul 뒤 Add 패턴에서 q, k, v는 3개씩 묶임
    # attention block 당 3개 bias (q, k, v) + 1개 output projection + 2개 FFN
    # 실제로는 이름 패턴으로 구분해야 함

    # 이름에 패턴이 있는지 확인
    has_qkv_pattern = any("q_proj" in b or "k_proj" in b for b in bias_768)

    if has_qkv_pattern:
        q_biases = [b for b in bias_768 if "q_proj" in b]
        k_biases = [b for b in bias_768 if "k_proj" in b]
        v_biases = [b for b in bias_768 if "v_proj" in b]
        return list(zip(q_biases, k_biases, v_biases))

    # 이름 패턴이 없으면 순서로 추정 (위험하지만 시도)
    # wav2vec2에서 attention block당 768-dim bias: q, k, v, out_proj, fc1 관련
    # 실제 구조를 봐야 함
    print(f"  Found {len(bias_768)} 768-dim biases (no q/k/v naming pattern)")
    print(f"  First 20: {bias_768[:20]}")
    return None


def run_fp32_inference(onnx_path, input_data):
    """ONNX FP32 추론"""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: input_data})
    return result[0]


def ctc_greedy_decode(logits, vocab):
    """CTC greedy decode"""
    tokens = np.argmax(logits, axis=-1)
    if len(tokens.shape) > 1:
        tokens = tokens[0]

    # 연속 중복 제거
    prev = -1
    decoded = []
    for t in tokens:
        if t != prev:
            if t != 0:  # blank=0
                decoded.append(t)
            prev = t

    text = ""
    for t in decoded:
        if t < len(vocab):
            text += vocab[t]
    return text


def main():
    print("=" * 70)
    print("Korean Wav2Vec2 k_proj Bias Injection Experiment")
    print("=" * 70)

    # 모델 로드
    print("\nLoading Korean model...")
    ko_model = onnx.load(KO_ONNX)
    print(f"  {KO_ONNX}")

    print("Loading English model...")
    en_model = onnx.load(EN_ONNX)
    print(f"  {EN_ONNX}")

    # k_proj bias 찾기
    print("\n--- Identifying k_proj biases ---")
    ko_qkv = identify_qkv_bias_groups(ko_model)
    en_qkv = identify_qkv_bias_groups(en_model)

    if ko_qkv:
        ko_k_biases = [g[1] for g in ko_qkv]
        print(f"\nKorean k_proj biases ({len(ko_k_biases)}): {ko_k_biases[:3]}...")
    else:
        # 이름 패턴 없으면 graph trace로 찾기
        ko_k_biases = find_kproj_bias_by_graph_trace(ko_model)
        print(f"\nKorean 768-dim biases (graph trace): {len(ko_k_biases)} found")

    if en_qkv:
        en_k_biases = [g[1] for g in en_qkv]
        print(f"English k_proj biases ({len(en_k_biases)}): {en_k_biases[:3]}...")
    else:
        en_k_biases = find_kproj_bias_by_graph_trace(en_model)
        print(f"English 768-dim biases (graph trace): {len(en_k_biases)} found")

    # bias 통계
    if ko_qkv and en_qkv:
        analyze_biases(ko_model, ko_k_biases, "Korean")
        analyze_biases(en_model, en_k_biases, "English")

    # 테스트 입력 준비
    if os.path.exists(TEST_INPUT_NPY):
        test_input = np.load(TEST_INPUT_NPY)
        print(f"\nTest input shape: {test_input.shape}")
    else:
        # 랜덤 입력으로 대체
        print(f"\nCalibration npy not found, using random input")
        test_input = np.random.randn(1, 48000).astype(np.float32) * 0.1

    # 한국어 vocab 로드
    vocab_path = os.path.join(BASE_DIR, "vocab.json")
    if os.path.exists(vocab_path):
        import json
        with open(vocab_path) as f:
            vocab_dict = json.load(f)
        vocab = [""] * (max(vocab_dict.values()) + 1)
        for char, idx in vocab_dict.items():
            vocab[idx] = char
        print(f"Korean vocab: {len(vocab)} tokens")
    else:
        vocab = None

    # === 실험 0: 원본 한국어 FP32 추론 ===
    print("\n" + "=" * 70)
    print("Experiment 0: Original Korean FP32")
    print("=" * 70)

    orig_output = run_fp32_inference(KO_ONNX, test_input)
    print(f"  Output shape: {orig_output.shape}")
    print(f"  Logit range: [{orig_output.min():.2f}, {orig_output.max():.2f}]")

    top2 = np.sort(orig_output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]
    print(f"  Top1-Top2 gap: mean={gaps.mean():.3f}, min={gaps.min():.3f}, max={gaps.max():.3f}")

    if vocab:
        text = ctc_greedy_decode(orig_output[0], vocab)
        print(f"  Decoded: {text}")

    # === 각 실험에서 사용할 수정 함수들 ===

    if not (ko_qkv and en_qkv):
        print("\n!!! Cannot identify q/k/v bias groups. Trying alternative approach...")
        # 모든 initializer 이름 출력 (bias 관련)
        print("\nKorean model initializer names containing 'bias':")
        for init in ko_model.graph.initializer:
            if "bias" in init.name.lower() or "Bias" in init.name:
                print(f"  {init.name}: shape={list(init.dims)}")

        print("\nEnglish model initializer names containing 'bias':")
        for init in en_model.graph.initializer:
            if "bias" in init.name.lower() or "Bias" in init.name:
                print(f"  {init.name}: shape={list(init.dims)}")
        return

    n_layers = min(len(ko_k_biases), len(en_k_biases))
    print(f"\nMatched {n_layers} layers for bias injection")

    # === 실험 1: 영어 k_proj bias 직접 이식 ===
    print("\n" + "=" * 70)
    print("Experiment 1: Inject English k_proj biases into Korean model")
    print("=" * 70)

    ko_model_exp1 = onnx.load(KO_ONNX)
    for i in range(n_layers):
        en_bias = get_initializer(en_model, en_k_biases[i])
        set_initializer(ko_model_exp1, ko_k_biases[i], en_bias)
        if i < 3:
            print(f"  Layer {i}: injected en bias (mean_abs={np.mean(np.abs(en_bias)):.4f})")

    exp1_path = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_en_kbias_nopad10_opset12_sim.onnx")
    onnx.save(ko_model_exp1, exp1_path)
    print(f"  Saved: {exp1_path}")
    print(f"  Size: {os.path.getsize(exp1_path) / 1e6:.1f}MB")

    exp1_output = run_fp32_inference(exp1_path, test_input)
    print(f"  Logit range: [{exp1_output.min():.2f}, {exp1_output.max():.2f}]")
    top2 = np.sort(exp1_output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]
    print(f"  Top1-Top2 gap: mean={gaps.mean():.3f}")
    if vocab:
        text = ctc_greedy_decode(exp1_output[0], vocab)
        print(f"  Decoded: {text}")

    # === 실험 2: 한국어 k_proj bias를 영어 수준으로 스케일업 ===
    print("\n" + "=" * 70)
    print("Experiment 2: Scale up Korean k_proj biases (×100)")
    print("=" * 70)

    ko_model_exp2 = onnx.load(KO_ONNX)
    for i in range(n_layers):
        ko_bias = get_initializer(ko_model_exp2, ko_k_biases[i])
        # 한국어 mean_abs=0.14, 영어 mean_abs=15.11 → ×108 ≈ ×100
        scaled_bias = ko_bias * 100.0
        set_initializer(ko_model_exp2, ko_k_biases[i], scaled_bias)
        if i < 3:
            print(f"  Layer {i}: scaled bias ×100 (mean_abs={np.mean(np.abs(scaled_bias)):.4f})")

    exp2_path = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_kbias_x100_nopad10_opset12_sim.onnx")
    onnx.save(ko_model_exp2, exp2_path)
    print(f"  Saved: {exp2_path}")

    exp2_output = run_fp32_inference(exp2_path, test_input)
    print(f"  Logit range: [{exp2_output.min():.2f}, {exp2_output.max():.2f}]")
    top2 = np.sort(exp2_output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]
    print(f"  Top1-Top2 gap: mean={gaps.mean():.3f}")
    if vocab:
        text = ctc_greedy_decode(exp2_output[0], vocab)
        print(f"  Decoded: {text}")

    # === 실험 3: 영어 k_proj bias + q_proj bias 둘 다 이식 ===
    print("\n" + "=" * 70)
    print("Experiment 3: Inject English k_proj + q_proj biases")
    print("=" * 70)

    ko_model_exp3 = onnx.load(KO_ONNX)
    ko_q_biases = [g[0] for g in ko_qkv]
    en_q_biases = [g[0] for g in en_qkv]

    for i in range(n_layers):
        en_k_bias = get_initializer(en_model, en_k_biases[i])
        en_q_bias = get_initializer(en_model, en_q_biases[i])
        set_initializer(ko_model_exp3, ko_k_biases[i], en_k_bias)
        set_initializer(ko_model_exp3, ko_q_biases[i], en_q_bias)

    exp3_path = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_en_qkbias_nopad10_opset12_sim.onnx")
    onnx.save(ko_model_exp3, exp3_path)
    print(f"  Saved: {exp3_path}")

    exp3_output = run_fp32_inference(exp3_path, test_input)
    print(f"  Logit range: [{exp3_output.min():.2f}, {exp3_output.max():.2f}]")
    top2 = np.sort(exp3_output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]
    print(f"  Top1-Top2 gap: mean={gaps.mean():.3f}")
    if vocab:
        text = ctc_greedy_decode(exp3_output[0], vocab)
        print(f"  Decoded: {text}")

    # === 실험 4: SmoothQuant - activation range를 weight로 이전 ===
    print("\n" + "=" * 70)
    print("Experiment 4: SmoothQuant-style — scale Q,K weights to reduce Q@K^T range")
    print("=" * 70)

    # SmoothQuant: W_new = W * diag(s), X_new = X * diag(1/s)
    # 여기서는 K weight만 축소하여 Q@K^T 범위를 줄이되,
    # K bias도 같이 축소해서 일관성 유지

    ko_model_exp4 = onnx.load(KO_ONNX)

    # Q, K weight 이름도 찾아야 함
    # MatMul 노드에서 weight를 찾기
    init_names = {init.name for init in ko_model_exp4.graph.initializer}

    # k_proj MatMul의 weight 찾기
    # k_proj.bias를 사용하는 Add 노드의 입력을 추적
    k_weight_names = []
    for layer_idx in range(n_layers):
        k_bias_name = ko_k_biases[layer_idx]
        # k_bias를 사용하는 Add 노드 찾기
        for node in ko_model_exp4.graph.node:
            if node.op_type == "Add" and k_bias_name in node.input:
                # 이 Add의 다른 입력은 MatMul의 출력
                matmul_output = [inp for inp in node.input if inp != k_bias_name][0]
                # 이 출력을 생성하는 MatMul 노드 찾기
                for node2 in ko_model_exp4.graph.node:
                    if node2.op_type == "MatMul" and matmul_output in node2.output:
                        # MatMul의 두 입력 중 initializer인 것이 weight
                        for inp in node2.input:
                            if inp in init_names:
                                k_weight_names.append(inp)
                                break
                        break
                break

    print(f"  Found {len(k_weight_names)} k_proj weights")

    # K weight를 1/4로 축소 (Q@K^T 범위 4배 감소)
    scale_factor = 0.25
    for i in range(min(len(k_weight_names), n_layers)):
        k_weight = get_initializer(ko_model_exp4, k_weight_names[i])
        k_bias = get_initializer(ko_model_exp4, ko_k_biases[i])

        set_initializer(ko_model_exp4, k_weight_names[i], k_weight * scale_factor)
        set_initializer(ko_model_exp4, ko_k_biases[i], k_bias * scale_factor)

        if i < 3:
            orig_range = np.max(np.abs(k_weight))
            print(f"  Layer {i}: K weight max_abs {orig_range:.4f} → {orig_range*scale_factor:.4f}")

    exp4_path = os.path.join(BASE_DIR, "wav2vec2_ko_base_3s_kscale025_nopad10_opset12_sim.onnx")
    onnx.save(ko_model_exp4, exp4_path)
    print(f"  Saved: {exp4_path}")

    exp4_output = run_fp32_inference(exp4_path, test_input)
    print(f"  Logit range: [{exp4_output.min():.2f}, {exp4_output.max():.2f}]")
    top2 = np.sort(exp4_output[0], axis=-1)[:, -2:]
    gaps = top2[:, -1] - top2[:, -2]
    print(f"  Top1-Top2 gap: mean={gaps.mean():.3f}")
    if vocab:
        text = ctc_greedy_decode(exp4_output[0], vocab)
        print(f"  Decoded: {text}")

    # === 요약 ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<40} {'Logit Range':>12} {'Top1-2 Gap':>12} {'Decoded':>20}")
    print("-" * 90)

    experiments = [
        ("0: Original Korean", orig_output),
        ("1: EN k_proj bias injection", exp1_output),
        ("2: KO k_proj bias ×100", exp2_output),
        ("3: EN q+k_proj bias injection", exp3_output),
        ("4: K weight ×0.25 (smooth)", exp4_output),
    ]

    for name, output in experiments:
        logit_range = output.max() - output.min()
        top2 = np.sort(output[0], axis=-1)[:, -2:]
        gap = (top2[:, -1] - top2[:, -2]).mean()
        if vocab:
            text = ctc_greedy_decode(output[0], vocab)
            text_short = text[:18] + "..." if len(text) > 18 else text
        else:
            text_short = "N/A"
        print(f"  {name:<38} {logit_range:>10.2f}   {gap:>10.3f}   {text_short:>20}")

    # 가장 유망한 모델 파일 목록 출력
    print("\n--- Generated ONNX files ---")
    for path in [exp1_path, exp2_path, exp3_path, exp4_path]:
        if os.path.exists(path):
            print(f"  {os.path.basename(path)} ({os.path.getsize(path)/1e6:.1f}MB)")


if __name__ == "__main__":
    main()

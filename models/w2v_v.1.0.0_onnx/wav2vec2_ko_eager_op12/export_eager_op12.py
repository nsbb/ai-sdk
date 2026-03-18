#!/usr/bin/env python3
"""
Korean wav2vec2 → EN-identical ONNX (eager attention, opset 12)

기존 KO ONNX는 SDPA + opset 14로 export되어 영어와 구조가 다름 (1306 vs 957 nodes).
eager + opset 12로 재변환하면 영어와 100% 동일한 구조 (957 nodes).

Usage:
    python3 export_eager_op12.py [--input-length 48000] [--model-id ...]
"""
import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-length", type=int, default=48000, help="Input samples (48000=3s, 80000=5s)")
    parser.add_argument("--model-id", type=str, default=None, help="HF model ID or local path")
    args = parser.parse_args()

    INPUT_LENGTH = args.input_length
    duration_s = INPUT_LENGTH / 16000

    # Try local path first, then HuggingFace
    local_path = os.path.join(os.path.dirname(__file__), "..", "wav2vec2_ko_base_3s")
    if args.model_id:
        model_id = args.model_id
    elif os.path.exists(os.path.join(local_path, "config.json")):
        model_id = local_path
    else:
        model_id = "Kkonjeong/wav2vec2-base-korean"

    print(f"Model: {model_id}")
    print(f"Input: [1, {INPUT_LENGTH}] ({duration_s:.1f}s @ 16kHz)")

    import torch
    from transformers import Wav2Vec2ForCTC

    # 핵심: eager attention 강제 → SDPA 비활성화 → opset 12 호환
    model = Wav2Vec2ForCTC.from_pretrained(model_id, attn_implementation="eager")
    model.eval()

    print(f"Config attn_implementation: {model.config._attn_implementation}")
    print(f"Vocab size: {model.config.vocab_size}")

    dummy = torch.randn(1, INPUT_LENGTH)
    with torch.no_grad():
        out = model(dummy)
        logits = out.logits
        output_seq_len = logits.shape[1]
        vocab_size = logits.shape[2]
    print(f"Output: [1, {output_seq_len}, {vocab_size}]")

    # ONNX export (opset 12 = EN과 동일)
    output_name = f"wav2vec2_ko_eager_op12_{int(duration_s)}s.onnx"
    print(f"\nExporting {output_name} (opset 12, eager attention)...")

    torch.onnx.export(
        model, dummy, output_name,
        input_names=["input_values"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
    )

    # Shape 고정 (Acuity Toolkit용)
    import onnx
    m = onnx.load(output_name)
    for inp in m.graph.input:
        if inp.name == "input_values":
            inp.type.tensor_type.shape.dim[0].dim_value = 1
            inp.type.tensor_type.shape.dim[1].dim_value = INPUT_LENGTH
    for out_node in m.graph.output:
        if out_node.name == "logits":
            out_node.type.tensor_type.shape.dim[0].dim_value = 1
            out_node.type.tensor_type.shape.dim[1].dim_value = output_seq_len
            out_node.type.tensor_type.shape.dim[2].dim_value = vocab_size
    onnx.save(m, output_name)

    # 검증
    from collections import Counter
    m = onnx.load(output_name)
    ops = Counter(n.op_type for n in m.graph.node)
    print(f"\nVerification:")
    print(f"  Nodes: {len(m.graph.node)} (EN=957)")
    print(f"  Opset: {m.opset_import[0].version}")
    print(f"  MatMul={ops['MatMul']}, Reshape={ops['Reshape']}, Transpose={ops['Transpose']}, Softmax={ops['Softmax']}")
    print(f"  Shape={ops.get('Shape',0)}, Cast={ops.get('Cast',0)} (should be 1, 0)")
    print(f"  File: {os.path.getsize(output_name)/1e6:.1f} MB")

    # ONNX Runtime 검증
    import onnxruntime as ort
    sess = ort.InferenceSession(output_name)
    ort_out = sess.run(None, {"input_values": dummy.numpy()})[0]
    max_diff = np.abs(logits.numpy() - ort_out).max()
    print(f"  PyTorch vs ONNX max diff: {max_diff:.8f}")

    # I/O shape 확인
    for i in sess.get_inputs():
        print(f"  Input: {i.name} {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  Output: {o.name} {o.shape} {o.type}")

    print(f"\nDone: {output_name}")

if __name__ == "__main__":
    main()

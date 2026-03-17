#!/usr/bin/env python3
"""
Create a CNN-only variant of the Korean wav2vec2 model.

Architecture:
  Original: input_values -> CNN feature_extractor (7 conv) -> feature_projection (512->768)
            -> Transformer encoder (12 layers) -> lm_head (768->56) -> logits

  CNN-only: input_values -> CNN feature_extractor (7 conv) -> feature_projection (512->768)
            -> lm_head (768->56) -> logits

The Transformer encoder is removed entirely. The feature_projection output (768-dim)
is connected directly to lm_head.
"""

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np
import os

WORK_DIR = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_ko_base_3s"
INPUT_MODEL = os.path.join(WORK_DIR, "wav2vec2_ko_base_3s.onnx")
OUTPUT_MODEL = os.path.join(WORK_DIR, "wav2vec2_ko_base_3s_cnn_only.onnx")

print(f"Loading model from {INPUT_MODEL}...")
model = onnx.load(INPUT_MODEL)
graph = model.graph

# === Identify the nodes to keep ===
# Nodes 0-87: CNN feature_extractor + feature_projection (output: 768-dim)
# Nodes 1304-1305: lm_head (768->56)
# Skip nodes 88-1303: pos_conv_embed + encoder layer_norm + 12 Transformer layers

# The feature_projection output tensor name:
feat_proj_output = "/wav2vec2/feature_projection/projection/Add_output_0"

# The lm_head input currently comes from encoder output:
# Node 1304 (lm_head/MatMul): inputs = [encoder_output, weight]
# We need to rewire it to take feat_proj_output instead.

print("Building CNN-only model...")

# Collect nodes to keep: 0-87 (CNN + feature_projection) + 1304-1305 (lm_head)
keep_nodes = list(graph.node[:88])  # nodes 0 through 87

# Create modified lm_head nodes that take feature_projection output
lm_head_matmul = onnx.NodeProto()
lm_head_matmul.CopyFrom(graph.node[1304])
# Replace encoder output with feature_projection output
lm_head_matmul.ClearField('input')
lm_head_matmul.input.append(feat_proj_output)       # was: encoder layer11 final_layer_norm output
lm_head_matmul.input.append('onnx::MatMul_1910')    # lm_head weight (unchanged)
keep_nodes.append(lm_head_matmul)

lm_head_add = onnx.NodeProto()
lm_head_add.CopyFrom(graph.node[1305])
# This node is fine as-is (inputs: lm_head.bias + lm_head/MatMul_output_0)
keep_nodes.append(lm_head_add)

print(f"  Original nodes: {len(graph.node)}")
print(f"  CNN-only nodes: {len(keep_nodes)}")

# Collect initializers to keep (only those referenced by kept nodes)
needed_initializer_names = set()
for node in keep_nodes:
    for inp in node.input:
        needed_initializer_names.add(inp)

kept_initializers = []
for init in graph.initializer:
    if init.name in needed_initializer_names:
        kept_initializers.append(init)

print(f"  Original initializers: {len(graph.initializer)}")
print(f"  CNN-only initializers: {len(kept_initializers)}")

# Build new graph
new_graph = helper.make_graph(
    nodes=keep_nodes,
    name="wav2vec2_ko_cnn_only",
    inputs=list(graph.input),   # Same input: input_values [1, 48000]
    outputs=list(graph.output), # Same output: logits [1, 149, 56]
    initializer=kept_initializers,
)

# Build new model
new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
new_model.ir_version = model.ir_version

# Validate
print("\nValidating model...")
try:
    onnx.checker.check_model(new_model)
    print("  Model validation passed!")
except Exception as e:
    print(f"  Validation warning: {e}")
    print("  (This may be OK for large models)")

# Save
print(f"\nSaving to {OUTPUT_MODEL}...")
onnx.save(new_model, OUTPUT_MODEL)

# Report sizes
orig_size = os.path.getsize(INPUT_MODEL)
new_size = os.path.getsize(OUTPUT_MODEL)
print(f"\n=== Size comparison ===")
print(f"  Original model: {orig_size:,} bytes ({orig_size/1024/1024:.1f} MB)")
print(f"  CNN-only model: {new_size:,} bytes ({new_size/1024/1024:.1f} MB)")
print(f"  Reduction: {(1 - new_size/orig_size)*100:.1f}%")

# Verify with onnxruntime
print("\n=== Verifying with onnxruntime ===")
import onnxruntime as ort

sess = ort.InferenceSession(OUTPUT_MODEL)
inp_info = sess.get_inputs()[0]
out_info = sess.get_outputs()[0]
print(f"  Input:  {inp_info.name} shape={inp_info.shape} type={inp_info.type}")
print(f"  Output: {out_info.name} shape={out_info.shape} type={out_info.type}")

# Run with dummy input
dummy = np.random.randn(1, 48000).astype(np.float32)
result = sess.run(None, {inp_info.name: dummy})
print(f"  Output shape: {result[0].shape}")
print(f"  Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")

# Compare with original model on test input
if os.path.exists(os.path.join(WORK_DIR, "test_audio.npy")):
    print("\n=== Comparing with original on test_audio.npy ===")
    test_audio = np.load(os.path.join(WORK_DIR, "test_audio.npy"))
    print(f"  test_audio shape: {test_audio.shape}")

    # CNN-only result
    cnn_result = sess.run(None, {inp_info.name: test_audio})[0]
    print(f"  CNN-only output shape: {cnn_result.shape}")
    print(f"  CNN-only output range: [{cnn_result.min():.4f}, {cnn_result.max():.4f}]")

    # Original model result
    orig_sess = ort.InferenceSession(INPUT_MODEL)
    orig_result = orig_sess.run(None, {inp_info.name: test_audio})[0]
    print(f"  Original output shape: {orig_result.shape}")
    print(f"  Original output range: [{orig_result.min():.4f}, {orig_result.max():.4f}]")

    # CTC decode both
    cnn_tokens = np.argmax(cnn_result[0], axis=-1)
    orig_tokens = np.argmax(orig_result[0], axis=-1)

    # Simple CTC greedy decode
    def ctc_greedy(tokens, blank_id=0):
        result = []
        prev = blank_id
        for t in tokens:
            if t != blank_id and t != prev:
                result.append(int(t))
            prev = t
        return result

    cnn_decoded = ctc_greedy(cnn_tokens)
    orig_decoded = ctc_greedy(orig_tokens)

    # Load vocab
    import json
    vocab_path = os.path.join(WORK_DIR, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        inv_vocab = {v: k for k, v in vocab.items()}
        cnn_text = "".join([inv_vocab.get(t, "?") for t in cnn_decoded])
        orig_text = "".join([inv_vocab.get(t, "?") for t in orig_decoded])
        print(f"\n  CNN-only decoded: '{cnn_text}'")
        print(f"  Original decoded: '{orig_text}'")

print("\nDone!")

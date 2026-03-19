#!/usr/bin/env python3
"""
fix_negative_gather_v2.py

Fixes negative Gather indices in zipformer_encoder_folded4_with_states_v2.onnx.
Key insight: each Gather node gets its OWN new constant node, even if multiple
Gathers shared the same old index constant. This avoids cross-contamination
when different Gathers need different positive values.
"""

import sys
import copy
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, TensorProto

MODEL_IN  = "zipformer_encoder_folded4_with_states_v2.onnx"
MODEL_OUT = "zipformer_encoder_folded4_with_states_v6.onnx"

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load model
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_IN} …")
model = onnx.load(MODEL_IN)
graph = model.graph

# Build initializer (constant) lookup: name → numpy array
init_map = {}
for init in graph.initializer:
    init_map[init.name] = numpy_helper.to_array(init)

# Build node output → node map for constant-value tracing
node_output_map = {}  # output_name → node
for node in graph.node:
    for out in node.output:
        node_output_map[out] = node

def get_constant_value(name):
    """Return numpy scalar/array if 'name' is a constant (initializer or Constant node),
    else None."""
    if name in init_map:
        return init_map[name]
    if name in node_output_map:
        nd = node_output_map[name]
        if nd.op_type == "Constant":
            for attr in nd.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 2. Find all Gather nodes with negative constant indices
# ──────────────────────────────────────────────────────────────────────────────
negative_gathers = []  # list of dicts

for node in graph.node:
    if node.op_type != "Gather":
        continue
    if len(node.input) < 2:
        continue

    idx_name = node.input[1]
    idx_val = get_constant_value(idx_name)
    if idx_val is None:
        continue  # dynamic index, skip

    idx_scalar = int(idx_val.flat[0]) if idx_val.size == 1 else None
    if idx_scalar is None:
        if np.any(idx_val < 0):
            print(f"  WARNING: Gather '{node.name}' has array of negative indices — skipping.")
        continue

    if idx_scalar >= 0:
        continue  # already positive

    # Get axis attribute (default 0)
    axis = 0
    for attr in node.attribute:
        if attr.name == "axis":
            axis = int(attr.i)

    data_name = node.input[0]
    negative_gathers.append({
        "node_name": node.name,
        "data_name": data_name,
        "idx_name":  idx_name,
        "neg_idx":   idx_scalar,
        "axis":      axis,
        "node":      node,
    })

print(f"\nFound {len(negative_gathers)} Gather node(s) with negative constant indices.")

# Diagnose shared index constants
idx_name_to_gathers = {}
for g in negative_gathers:
    idx_name_to_gathers.setdefault(g["idx_name"], []).append(g["node_name"])

print("\nIndex constant sharing analysis:")
for idx_name, gather_names in idx_name_to_gathers.items():
    print(f"  Constant '{idx_name}' used by {len(gather_names)} Gather(s): {gather_names}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Run ONNX Runtime with intermediate outputs to get data tensor shapes
# ──────────────────────────────────────────────────────────────────────────────
needed_tensors = list({g["data_name"] for g in negative_gathers})

# Add them as extra graph outputs
tmp_model = copy.deepcopy(model)
existing_outputs = {o.name for o in tmp_model.graph.output}

try:
    tmp_model_shaped = onnx.shape_inference.infer_shapes(tmp_model)
except Exception as e:
    print(f"  shape inference warning: {e}")
    tmp_model_shaped = tmp_model

vi_map = {}
for vi in tmp_model_shaped.graph.value_info:
    vi_map[vi.name] = vi
for vi in tmp_model_shaped.graph.input:
    vi_map[vi.name] = vi
for vi in tmp_model_shaped.graph.output:
    vi_map[vi.name] = vi

for tname in needed_tensors:
    if tname not in existing_outputs:
        if tname in vi_map:
            tmp_model.graph.output.append(vi_map[tname])
        else:
            tmp_model.graph.output.append(onnx.helper.make_tensor_value_info(
                tname, TensorProto.FLOAT, None))

# Build inference feed
dtype_map = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.INT64:  np.int64,
    TensorProto.INT32:  np.int32,
    TensorProto.BOOL:   np.bool_,
    TensorProto.DOUBLE: np.float64,
}
CONCRETE_X_SHAPE = [1, 39, 80]

feed = {}
for inp in model.graph.input:
    t = inp.type.tensor_type
    shape = []
    for dim in t.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append(1)
    dtype = dtype_map.get(t.elem_type, np.float32)
    if inp.name == "x":
        arr = np.random.randn(*CONCRETE_X_SHAPE).astype(dtype)
    else:
        arr = np.zeros(shape, dtype=dtype)
    feed[inp.name] = arr

print("\nRunning ONNX Runtime to capture intermediate shapes …")
sess_opts = ort.SessionOptions()
sess_opts.log_severity_level = 3
tmp_bytes = tmp_model.SerializeToString()
try:
    sess_tmp = ort.InferenceSession(tmp_bytes, sess_options=sess_opts,
                                    providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess_tmp.get_outputs()]
    results = sess_tmp.run(output_names, feed)
    result_map = dict(zip(output_names, results))
    print("  Inference succeeded.")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

tensor_shapes = {}
for tname in needed_tensors:
    if tname in result_map:
        tensor_shapes[tname] = result_map[tname].shape
    else:
        print(f"  WARNING: '{tname}' not in inference outputs.")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Compute positive indices and print summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*95)
print(f"{'Gather Node':<45} {'Data Tensor':<35} {'Neg':>4} {'Shape':>22} {'Axis':>4} {'Pos':>4}")
print("="*95)

for g in negative_gathers:
    tname = g["data_name"]
    shape = tensor_shapes.get(tname, None)
    neg   = g["neg_idx"]
    axis  = g["axis"]
    if shape is not None and axis < len(shape):
        dim_size = shape[axis]
        pos = dim_size + neg  # e.g. -1 → N-1
    else:
        pos = None
    g["pos_idx"] = pos
    g["data_shape"] = shape

    shape_str = str(tuple(shape)) if shape is not None else "unknown"
    print(f"{g['node_name']:<45} {tname:<35} {neg:>4} {shape_str:>22} {axis:>4} {str(pos):>4}")

print("="*95)

# Check if there are any shared constant names that need DIFFERENT positive values
problem_found = False
for idx_name, glist in idx_name_to_gathers.items():
    pos_vals = [g["pos_idx"] for g in negative_gathers if g["idx_name"] == idx_name]
    if len(set(pos_vals)) > 1:
        print(f"\n  *** CONFLICT: constant '{idx_name}' needs different positive values: {pos_vals}")
        problem_found = True

# ──────────────────────────────────────────────────────────────────────────────
# 5. Build fixed model — each Gather gets its own unique constant node
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nBuilding fixed model → {MODEL_OUT} …")
fixed_model = copy.deepcopy(model)
fixed_graph = fixed_model.graph

# Map node name → node object in fixed model
fixed_node_map = {nd.name: nd for nd in fixed_graph.node}

new_const_nodes = []
# Keep counter for unique naming
counter = 0

for g in negative_gathers:
    pos = g["pos_idx"]
    if pos is None:
        print(f"  SKIP (no shape): {g['node_name']}")
        continue

    # Always create a unique constant per Gather node (never reuse)
    new_const_name = f"__gather_pos_idx_{counter}"
    counter += 1

    const_tensor = numpy_helper.from_array(
        np.array(pos, dtype=np.int64), name=new_const_name + "_val")
    const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[new_const_name],
        name=new_const_name + "_node",
        value=const_tensor,
    )
    new_const_nodes.append(const_node)

    # Patch the Gather node's index input
    nd = fixed_node_map[g["node_name"]]
    nd.input[1] = new_const_name
    print(f"  Patched {g['node_name']}: index {g['neg_idx']} → {pos}  (const: {new_const_name})")

# Insert new Constant nodes at the beginning of the graph
for nd in reversed(new_const_nodes):
    fixed_graph.node.insert(0, nd)

onnx.save(fixed_model, MODEL_OUT)
print(f"\n  Saved: {MODEL_OUT}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Verify fixed model
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nVerifying fixed model …")
fixed_model_loaded = onnx.load(MODEL_OUT)
onnx.checker.check_model(fixed_model_loaded)
print("  onnx.checker.check_model: PASS")

sess_orig  = ort.InferenceSession(MODEL_IN,  sess_options=sess_opts,
                                  providers=["CPUExecutionProvider"])
sess_fixed = ort.InferenceSession(MODEL_OUT, sess_options=sess_opts,
                                  providers=["CPUExecutionProvider"])

feed_orig  = {k: v for k, v in feed.items() if k in {i.name for i in sess_orig.get_inputs()}}
feed_fixed = {k: v for k, v in feed.items() if k in {i.name for i in sess_fixed.get_inputs()}}

out_orig  = sess_orig.run(None, feed_orig)
out_fixed = sess_fixed.run(None, feed_fixed)

print(f"  Number of outputs: orig={len(out_orig)}, fixed={len(out_fixed)}")
all_close = True
for i, (a, b) in enumerate(zip(out_orig, out_fixed)):
    if not np.allclose(a, b, atol=1e-5, rtol=1e-4):
        print(f"  Output [{i}] MISMATCH: max_diff={np.max(np.abs(a - b)):.4e}")
        all_close = False
    else:
        print(f"  Output [{i}]: OK  shape={a.shape}  max_diff={np.max(np.abs(a - b)):.2e}")

if all_close:
    print("\n✓ All outputs match. Fixed model is correct.")
else:
    print("\n✗ Some outputs differ — investigating …")

print(f"\nDone. Fixed model: {MODEL_OUT}")

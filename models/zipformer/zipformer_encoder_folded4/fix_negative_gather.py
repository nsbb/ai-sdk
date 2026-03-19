#!/usr/bin/env python3
"""
fix_negative_gather.py

Fixes negative Gather indices in zipformer_encoder_folded4_with_states_v2.onnx.
Replaces all constant -1 (or any negative) Gather index inputs with the
correct positive equivalents, determined by running ONNX Runtime inference
to get real intermediate tensor shapes.
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

    idx_scalar = int(idx_val.flat[0]) if idx_val.ndim == 0 else None
    if idx_scalar is None:
        # Array of indices — check if any are negative
        if np.any(idx_val < 0):
            print(f"  WARNING: Gather node '{node.name}' has array of indices with negatives; skipping for now.")
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
        "node":      node,        # reference
    })

print(f"\nFound {len(negative_gathers)} Gather node(s) with negative constant indices.")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Run ONNX Runtime with intermediate outputs to get data tensor shapes
# ──────────────────────────────────────────────────────────────────────────────
# Collect the data tensor names we need shapes for
needed_tensors = list({g["data_name"] for g in negative_gathers})

# Make a temporary model that also outputs those intermediate tensors
tmp_model = copy.deepcopy(model)
existing_outputs = {o.name for o in tmp_model.graph.output}

# Infer shapes first so value_info is populated
try:
    tmp_model_shaped = onnx.shape_inference.infer_shapes(tmp_model)
except Exception as e:
    print(f"  shape inference warning: {e}")
    tmp_model_shaped = tmp_model

# Collect value_info for needed tensors
vi_map = {}
for vi in tmp_model_shaped.graph.value_info:
    vi_map[vi.name] = vi
for vi in tmp_model_shaped.graph.input:
    vi_map[vi.name] = vi
for vi in tmp_model_shaped.graph.output:
    vi_map[vi.name] = vi

# Add needed tensors as extra graph outputs if not already there
for tname in needed_tensors:
    if tname not in existing_outputs:
        if tname in vi_map:
            tmp_model.graph.output.append(vi_map[tname])
        else:
            # Add without type info — ORT can still handle it
            tmp_model.graph.output.append(onnx.helper.make_tensor_value_info(
                tname, TensorProto.FLOAT, None))

# Build inference inputs
# Model inputs: name, shape
model_inputs_info = {}
for inp in model.graph.input:
    t = inp.type.tensor_type
    shape = []
    for dim in t.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(None)  # symbolic
        else:
            shape.append(None)
    dtype_map = {
        TensorProto.FLOAT:  np.float32,
        TensorProto.INT64:  np.int64,
        TensorProto.INT32:  np.int32,
        TensorProto.BOOL:   np.bool_,
        TensorProto.DOUBLE: np.float64,
    }
    elem_type = t.elem_type
    np_dtype = dtype_map.get(elem_type, np.float32)
    model_inputs_info[inp.name] = (shape, np_dtype)

# Determine concrete shapes for our inference run
# "x" → [1, 39, 80]
# State tensors → use model-declared shapes with 0-dims replaced by 1
CONCRETE_X_SHAPE = [1, 39, 80]

def make_feed(info_dict):
    feed = {}
    for name, (shape, dtype) in info_dict.items():
        if name == "x":
            concrete = CONCRETE_X_SHAPE
            arr = np.random.randn(*concrete).astype(dtype)
        else:
            concrete = []
            for s in shape:
                if s is None or s == 0:
                    concrete.append(1)
                else:
                    concrete.append(s)
            arr = np.zeros(concrete, dtype=dtype)
        feed[name] = arr
    return feed

print("\nBuilding inference feed …")
feed = make_feed(model_inputs_info)
print(f"  Input tensors: {list(feed.keys())}")
for k, v in feed.items():
    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

# Serialize tmp model and run
import io
tmp_bytes = tmp_model.SerializeToString()
print("\nRunning ONNX Runtime to capture intermediate shapes …")
try:
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress warnings
    sess = ort.InferenceSession(tmp_bytes, sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(output_names, feed)
    result_map = dict(zip(output_names, results))
    print("  Inference succeeded.")
except Exception as e:
    print(f"  ERROR during inference: {e}")
    sys.exit(1)

# Extract shapes for needed tensors
tensor_shapes = {}
for tname in needed_tensors:
    if tname in result_map:
        tensor_shapes[tname] = result_map[tname].shape
    else:
        print(f"  WARNING: '{tname}' not found in inference outputs.")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Compute positive indices and print summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'Gather Node':<45} {'Data Tensor':<30} {'Neg':>4} {'Shape':>20} {'Axis':>4} {'Pos':>4}")
print("="*90)

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
        dim_size = "?"
    g["pos_idx"] = pos
    g["data_shape"] = shape

    shape_str = str(tuple(shape)) if shape is not None else "unknown"
    print(f"{g['node_name']:<45} {tname:<30} {neg:>4} {shape_str:>20} {axis:>4} {str(pos):>4}")

print("="*90)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Build fixed model
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nBuilding fixed model → {MODEL_OUT} …")
fixed_model = copy.deepcopy(model)
fixed_graph = fixed_model.graph

# Map: old idx_name → new constant name (to reuse if same old name is shared)
replacement_map = {}  # old_idx_name → new_const_name

# We need to remap node.input[1] for affected Gather nodes.
# Build a lookup from node name to node in fixed_graph
fixed_node_map = {nd.name: nd for nd in fixed_graph.node}

new_nodes = []  # new Constant nodes to insert

for g in negative_gathers:
    pos = g["pos_idx"]
    if pos is None:
        print(f"  SKIP (no shape): {g['node_name']}")
        continue
    old_idx = g["idx_name"]
    if old_idx in replacement_map:
        new_idx_name = replacement_map[old_idx]
    else:
        new_idx_name = old_idx + "_pos"
        replacement_map[old_idx] = new_idx_name
        # Create new Constant node with positive index
        const_tensor = numpy_helper.from_array(
            np.array(pos, dtype=np.int64), name=new_idx_name + "_tensor")
        const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[new_idx_name],
            name=new_idx_name + "_node",
            value=const_tensor,
        )
        new_nodes.append(const_node)

    # Patch the Gather node's index input
    nd = fixed_node_map[g["node_name"]]
    nd.input[1] = new_idx_name

# Insert new Constant nodes at the beginning of the graph
for nd in reversed(new_nodes):
    fixed_graph.node.insert(0, nd)

# Save
onnx.save(fixed_model, MODEL_OUT)
print(f"  Saved: {MODEL_OUT}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Verify fixed model
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nVerifying fixed model …")
fixed_model_loaded = onnx.load(MODEL_OUT)
onnx.checker.check_model(fixed_model_loaded)
print("  onnx.checker.check_model: PASS")

# Run both original and fixed model, compare outputs
orig_output_names = [o.name for o in model.graph.output]

sess_orig = ort.InferenceSession(MODEL_IN, sess_options=sess_opts,
                                 providers=["CPUExecutionProvider"])
sess_fixed = ort.InferenceSession(MODEL_OUT, sess_options=sess_opts,
                                  providers=["CPUExecutionProvider"])

# Rebuild feed with only declared inputs (orig model)
feed_orig = {k: v for k, v in feed.items()
             if k in {i.name for i in sess_orig.get_inputs()}}
feed_fixed = {k: v for k, v in feed.items()
              if k in {i.name for i in sess_fixed.get_inputs()}}

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
    print("\n✗ Some outputs differ. Please investigate.")

print(f"\nDone. Fixed model saved to: {MODEL_OUT}")

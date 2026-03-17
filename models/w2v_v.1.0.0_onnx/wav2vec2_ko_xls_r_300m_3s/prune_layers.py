#!/usr/bin/env python3
"""
Prune XLS-R-300M Korean wav2vec2 ONNX model from 24 to 12 Transformer layers.

Removes encoder layers 12-23, reconnects layer 11's output to the
final encoder layer_norm -> lm_head classification head.

Key insight: weight tensors have opaque names like 'onnx::MatMul_3565'
so we must identify them by tracing which nodes consume them, not by name.

Input:  wav2vec2_ko_3s.onnx        (1.2GB, 24 layers)
Output: wav2vec2_ko_3s_12layers.onnx (~600MB, 12 layers)
"""

import onnx
from onnx import TensorProto
import re
import sys
import os

INPUT_MODEL = "wav2vec2_ko_3s.onnx"
OUTPUT_MODEL = "wav2vec2_ko_3s_12layers.onnx"

# Layers to remove: 12 through 23
REMOVE_LAYERS = set(range(12, 24))

# The last kept layer's output tensor name
LAST_KEPT_OUTPUT = "/wav2vec2/encoder/layers.11/Add_1_output_0"

# The removed layer's output that feeds into the post-encoder layer_norm
REPLACED_INPUT = "/wav2vec2/encoder/layers.23/Add_1_output_0"


def is_layer_node(node_name, layer_indices):
    """Check if a node belongs to one of the specified encoder layers."""
    m = re.search(r'/wav2vec2/encoder/layers\.(\d+)/', node_name)
    if m:
        return int(m.group(1)) in layer_indices
    return False


def main():
    print(f"Loading model: {INPUT_MODEL}")
    model = onnx.load(INPUT_MODEL)
    graph = model.graph

    orig_nodes = len(graph.node)
    orig_inits = len(graph.initializer)
    print(f"Original model: {orig_nodes} nodes, {orig_inits} initializers")

    # Step 1: Separate nodes into keep/remove
    nodes_to_remove = []
    nodes_to_keep = []
    removed_node_names = set()
    for node in graph.node:
        if is_layer_node(node.name, REMOVE_LAYERS):
            nodes_to_remove.append(node)
            removed_node_names.add(node.name)
        else:
            nodes_to_keep.append(node)

    print(f"\nNodes to remove: {len(nodes_to_remove)}")
    print(f"Nodes to keep: {len(nodes_to_keep)}")

    # Step 2: Collect ALL input tensor names consumed ONLY by removed nodes
    # Build usage map: tensor_name -> set of node names that consume it
    tensor_consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                if inp not in tensor_consumers:
                    tensor_consumers[inp] = set()
                tensor_consumers[inp].add(node.name)

    # Find initializers consumed EXCLUSIVELY by removed nodes
    init_names_to_remove = set()
    init_name_set = {init.name for init in graph.initializer}

    for node in nodes_to_remove:
        for inp in node.input:
            if inp in init_name_set:
                consumers = tensor_consumers.get(inp, set())
                # Only remove if ALL consumers are in the removed set
                if consumers.issubset(removed_node_names):
                    init_names_to_remove.add(inp)

    # Calculate size of removed initializers
    removed_size = 0
    for init in graph.initializer:
        if init.name in init_names_to_remove:
            removed_size += len(init.raw_data)

    print(f"\nInitializers to remove: {len(init_names_to_remove)} "
          f"({removed_size/1024/1024:.1f} MB)")

    # Step 3: Rewire - replace references to layer 23 output with layer 11 output
    rewire_count = 0
    for node in nodes_to_keep:
        new_inputs = []
        changed = False
        for inp in node.input:
            if inp == REPLACED_INPUT:
                new_inputs.append(LAST_KEPT_OUTPUT)
                changed = True
                rewire_count += 1
            else:
                new_inputs.append(inp)
        if changed:
            del node.input[:]
            node.input.extend(new_inputs)
            print(f"  Rewired: {node.name}")

    print(f"Rewired {rewire_count} input connections")

    # Step 4: Replace graph nodes
    del graph.node[:]
    graph.node.extend(nodes_to_keep)

    # Step 5: Remove initializers for pruned layers
    kept_inits = [init for init in graph.initializer
                  if init.name not in init_names_to_remove]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)

    # Step 6: Clean up value_info for removed intermediate tensors
    # Collect all tensor names produced by removed nodes
    removed_outputs = set()
    for node in nodes_to_remove:
        for out in node.output:
            removed_outputs.add(out)

    kept_value_info = [vi for vi in graph.value_info
                       if vi.name not in removed_outputs]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)

    # Step 6b: Remove orphaned initializers not consumed by any remaining node
    used_by_kept = set()
    for node in nodes_to_keep:
        for inp in node.input:
            used_by_kept.add(inp)

    orphaned = []
    kept_inits2 = []
    for init in graph.initializer:
        if init.name in used_by_kept:
            kept_inits2.append(init)
        else:
            orphaned.append(init.name)

    if orphaned:
        print(f"\nRemoving {len(orphaned)} orphaned initializers")
        del graph.initializer[:]
        graph.initializer.extend(kept_inits2)

    print(f"\nPruned model: {len(graph.node)} nodes, {len(graph.initializer)} initializers")
    print(f"Removed: {orig_nodes - len(graph.node)} nodes, "
          f"{orig_inits - len(graph.initializer)} initializers")

    # Step 7: Validate the graph connections
    print("\nValidating graph connections...")
    available_tensors = set()
    for inp in graph.input:
        available_tensors.add(inp.name)
    for init in graph.initializer:
        available_tensors.add(init.name)

    missing = []
    for node in graph.node:
        for inp in node.input:
            if inp and inp not in available_tensors:
                missing.append((node.name, inp))
        for out in node.output:
            available_tensors.add(out)

    if missing:
        print(f"WARNING: {len(missing)} missing tensor references!")
        for node_name, tensor_name in missing[:10]:
            print(f"  Node {node_name} needs: {tensor_name}")
    else:
        print("All tensor references are valid!")

    # Step 8: Verify output connects correctly
    output_name = graph.output[0].name
    # Trace back from output to find the last node producing it
    for node in reversed(list(graph.node)):
        if output_name in node.output:
            print(f"\nOutput '{output_name}' is produced by: {node.name}")
            break

    # Step 9: Save
    print(f"\nSaving pruned model to: {OUTPUT_MODEL}")
    onnx.save(model, OUTPUT_MODEL)

    # Check file size
    size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    orig_size_mb = os.path.getsize(INPUT_MODEL) / (1024 * 1024)
    print(f"Original size: {orig_size_mb:.1f} MB")
    print(f"Pruned size:   {size_mb:.1f} MB")
    print(f"Reduction:     {(1 - size_mb/orig_size_mb)*100:.1f}%")

    # Step 10: Run onnx checker (may fail for large models due to protobuf limits)
    print("\nRunning ONNX checker...")
    try:
        onnx.checker.check_model(OUTPUT_MODEL)
        print("ONNX check passed!")
    except Exception as e:
        err_str = str(e)
        if "2GB" in err_str or "protobuf" in err_str.lower():
            print("ONNX check skipped (model > 2GB protobuf limit) - this is expected")
        else:
            print(f"ONNX check warning: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()

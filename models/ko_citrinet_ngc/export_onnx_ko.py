#!/usr/bin/env python3
import argparse
import os
import types

import nemo.collections.asr as nemo_asr
import nemo.collections.asr.parts.submodules.jasper as jasper
import torch
import torch.nn as nn


def patch_squeeze_excite(model):
    def new_se_forward(self, x):
        scale = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        scale = self.fc(scale)
        scale = torch.sigmoid(scale)
        return x * scale

    for module in model.modules():
        if isinstance(module, jasper.SqueezeExcite):
            module.forward = types.MethodType(new_se_forward, module)


def patch_jasper_blocks(model):
    def new_jasper_block_forward(self, x):
        if isinstance(x, tuple):
            x = x[0]

        out = x
        for layer in self.mconv:
            if isinstance(layer, jasper.MaskedConv1d):
                out = layer.conv(out)
            elif isinstance(layer, nn.ModuleList):
                for sublayer in layer:
                    if isinstance(sublayer, jasper.MaskedConv1d):
                        out = sublayer.conv(out)
                    else:
                        out = sublayer(out)
            else:
                out = layer(out)

        if self.res is not None:
            for res_branch in self.res:
                res_out = x
                if isinstance(res_branch, nn.ModuleList):
                    for res_layer in res_branch:
                        if isinstance(res_layer, jasper.MaskedConv1d):
                            res_out = res_layer.conv(res_out)
                        else:
                            res_out = res_layer(res_out)
                elif isinstance(res_branch, jasper.MaskedConv1d):
                    res_out = res_branch.conv(res_out)
                else:
                    res_out = res_branch(res_out)
                out = out + res_out
        return out

    for module in model.modules():
        if isinstance(module, jasper.JasperBlock):
            module.forward = types.MethodType(new_jasper_block_forward, module)


def convert_to_2d(module, visited=None):
    if visited is None:
        visited = set()
    mid = id(module)
    if mid in visited:
        return
    visited.add(mid)

    for name, child in module.named_children():
        if isinstance(child, nn.Conv1d):
            new_conv = nn.Conv2d(
                child.in_channels,
                child.out_channels,
                kernel_size=(1, child.kernel_size[0]),
                stride=(1, child.stride[0]),
                padding=(0, child.padding[0]),
                dilation=(1, child.dilation[0]),
                groups=child.groups,
                bias=(child.bias is not None),
            )
            new_conv.weight.data = child.weight.data.unsqueeze(2)
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            setattr(module, name, new_conv)
        elif isinstance(child, nn.BatchNorm1d):
            new_bn = nn.BatchNorm2d(
                child.num_features, eps=child.eps, momentum=child.momentum
            )
            new_bn.weight.data = child.weight.data
            new_bn.bias.data = child.bias.data
            new_bn.running_mean.data = child.running_mean.data
            new_bn.running_var.data = child.running_var.data
            setattr(module, name, new_bn)
        elif isinstance(child, nn.Linear):
            new_fc = nn.Conv2d(child.in_features, child.out_features, 1)
            new_fc.weight.data = child.weight.data.unsqueeze(-1).unsqueeze(-1)
            if child.bias is not None:
                new_fc.bias.data = child.bias.data
            setattr(module, name, new_fc)
        else:
            convert_to_2d(child, visited=visited)


class SimplifiedCitrinet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enc = model.encoder.encoder
        self.dec = model.decoder.decoder_layers

    def forward(self, x):
        out = x
        for layer in self.enc:
            out = layer(out)
        for layer in self.dec:
            out = layer(out)
        return out


def fix_averagepool_to_global(onnx_path, output_path):
    import onnx
    from onnx import helper

    model = onnx.load(onnx_path)
    graph = model.graph
    new_nodes = []
    for node in graph.node:
        if node.op_type == "AveragePool":
            new_nodes.append(
                helper.make_node(
                    "GlobalAveragePool",
                    inputs=node.input,
                    outputs=node.output,
                    name=node.name,
                )
            )
        else:
            new_nodes.append(node)
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    onnx.save(model, output_path)


def load_model(model_name: str, model_file: str):
    if model_file:
        return nemo_asr.models.EncDecCTCModelBPE.restore_from(
            restore_path=model_file, map_location="cpu"
        )
    if not model_name:
        raise ValueError("Either --model-file or --model-name is required")
    return nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)


def maybe_dump_tokenizer(asr_model, out_dir: str):
    tok_dir = os.path.join(out_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)

    vocab_path = os.path.join(tok_dir, "vocab.txt")
    cfg_path = os.path.join(tok_dir, "model_config.yaml")
    try:
        labels = []
        if hasattr(asr_model, "tokenizer"):
            t = asr_model.tokenizer
            if hasattr(t, "tokenizer") and hasattr(t.tokenizer, "id_to_piece"):
                n = getattr(t, "vocab_size", None)
                if n is None:
                    n = asr_model.decoder.num_classes_with_blank - 1
                for i in range(n):
                    labels.append(t.tokenizer.id_to_piece(i))
            elif hasattr(t, "vocab"):
                labels = list(t.vocab)
        if labels:
            with open(vocab_path, "w", encoding="utf-8") as f:
                for x in labels:
                    f.write(str(x) + "\n")
        if hasattr(asr_model, "cfg"):
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(str(asr_model.cfg))
    except Exception as e:
        print(f"[WARN] tokenizer dump failed: {e}")


def main():
    p = argparse.ArgumentParser(description="Export Korean Citrinet to T527-friendly ONNX")
    p.add_argument("--model-name", default="", help="NeMo pretrained model name")
    p.add_argument("--model-file", default="", help="Local .nemo file path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--time-frames", type=int, default=300)
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    raw_onnx = os.path.join(out_dir, "citrinet_raw.onnx")
    sim_onnx = os.path.join(out_dir, "citrinet_sim.onnx")
    final_onnx = os.path.join(out_dir, "citrinet_npu.onnx")

    print("[1/6] load model")
    asr_model = load_model(args.model_name, args.model_file)
    asr_model.eval()

    print("[2/6] patch squeeze-excite")
    patch_squeeze_excite(asr_model)

    print("[3/6] patch jasper blocks")
    patch_jasper_blocks(asr_model)

    print("[4/6] convert 1d layers to 2d")
    convert_to_2d(asr_model)

    print("[5/6] export onnx")
    final_model = SimplifiedCitrinet(asr_model)
    final_model.eval()
    dummy_input = torch.randn(1, 80, 1, args.time_frames)
    torch.onnx.export(
        final_model,
        (dummy_input,),
        raw_onnx,
        opset_version=13,
        dynamo=False,
        input_names=["audio_signal"],
        output_names=["logits"],
        do_constant_folding=True,
    )

    input_shape_str = f'"audio_signal:1,80,1,{args.time_frames}"'
    ret = os.system(f"onnxsim {raw_onnx} {sim_onnx} --overwrite-input-shape {input_shape_str}")
    if ret != 0:
        import shutil

        print("[WARN] onnxsim failed, fallback to raw onnx")
        shutil.copy2(raw_onnx, sim_onnx)

    print("[6/6] convert AveragePool -> GlobalAveragePool")
    fix_averagepool_to_global(sim_onnx, final_onnx)

    import onnx

    m = onnx.load(final_onnx)
    onnx.checker.check_model(m)
    maybe_dump_tokenizer(asr_model, out_dir)

    print("[DONE]")
    print(f"onnx={final_onnx}")


if __name__ == "__main__":
    main()

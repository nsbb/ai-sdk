#!/usr/bin/env python3
import argparse
import os

import nemo.collections.asr as nemo_asr
import torch
import torch.nn as nn


class CitrinetNpuExportV2(nn.Module):
    """Export wrapper that keeps original encoder/decoder behavior with length input."""

    def __init__(self, asr_model):
        super().__init__()
        self.encoder = asr_model.encoder
        self.decoder = asr_model.decoder

    def forward(self, audio_signal, audio_signal_length):
        # input: [B,80,1,T] or [B,80,T]
        if audio_signal.dim() == 4:
            audio_signal = audio_signal.squeeze(2)

        enc, _ = self.encoder(audio_signal=audio_signal, length=audio_signal_length)
        # decoder output: [B,T,C]
        logits = self.decoder(encoder_output=enc)
        # NPU-friendly layout used by existing pipeline: [B,C,1,T]
        logits = logits.transpose(1, 2).unsqueeze(2)
        return logits


class CitrinetNpuExportV2FixedLen(nn.Module):
    """Export wrapper with fixed length (no int64 external input)."""

    def __init__(self, asr_model, time_frames: int):
        super().__init__()
        self.encoder = asr_model.encoder
        self.decoder = asr_model.decoder
        self.time_frames = int(time_frames)

    def forward(self, audio_signal):
        # input: [B,80,1,T] or [B,80,T]
        if audio_signal.dim() == 4:
            audio_signal = audio_signal.squeeze(2)

        # We export with fixed batch=1, so keep length as a literal constant tensor.
        # This avoids symbolic-shape handling that can stall ONNX tracing.
        ln = torch.tensor(
            [self.time_frames],
            dtype=torch.long,
            device=audio_signal.device,
        )
        enc, _ = self.encoder(audio_signal=audio_signal, length=ln)
        logits = self.decoder(encoder_output=enc)
        logits = logits.transpose(1, 2).unsqueeze(2)
        return logits


def load_model(model_name: str, model_file: str):
    if model_file:
        return nemo_asr.models.EncDecCTCModelBPE.restore_from(
            restore_path=model_file, map_location="cpu"
        )
    if not model_name:
        raise ValueError("Either --model-file or --model-name is required")
    return nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)


def main():
    p = argparse.ArgumentParser(
        description="Export Korean Citrinet to ONNX (v2, with length input preserved)"
    )
    p.add_argument("--model-name", default="", help="NeMo pretrained model name")
    p.add_argument("--model-file", default="", help="Local .nemo file path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--time-frames", type=int, default=300)
    p.add_argument("--onnx-name", default="citrinet_npu_v2.onnx")
    p.add_argument(
        "--with-length-input",
        action="store_true",
        help="export ONNX with explicit audio_signal_length input",
    )
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, args.onnx_name)

    print("[1/3] load model")
    asr_model = load_model(args.model_name, args.model_file)
    asr_model.eval()

    print("[2/3] export onnx")
    if args.with_length_input:
        final_model = CitrinetNpuExportV2(asr_model)
    else:
        final_model = CitrinetNpuExportV2FixedLen(asr_model, time_frames=args.time_frames)
    final_model.eval()

    dummy_x = torch.randn(1, 80, 1, args.time_frames)
    if args.with_length_input:
        dummy_len = torch.tensor([args.time_frames], dtype=torch.long)
        torch.onnx.export(
            final_model,
            (dummy_x, dummy_len),
            onnx_path,
            opset_version=13,
            dynamo=False,
            input_names=["audio_signal", "audio_signal_length"],
            output_names=["logits"],
            do_constant_folding=True,
            dynamic_axes=None,
        )
    else:
        torch.onnx.export(
            final_model,
            (dummy_x,),
            onnx_path,
            opset_version=13,
            dynamo=False,
            input_names=["audio_signal"],
            output_names=["logits"],
            do_constant_folding=True,
            dynamic_axes=None,
        )

    print("[3/3] validate onnx")
    import onnx

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    print("[DONE]")
    print(f"onnx={onnx_path}")


if __name__ == "__main__":
    main()

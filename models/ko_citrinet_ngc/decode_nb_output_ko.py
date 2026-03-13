#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import nemo.collections.asr as nemo_asr


def load_meta(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    out = next(iter(meta["Outputs"].values()))
    shape = [int(x) for x in out["shape"]]
    q = out.get("quantize", {})
    return shape, q


def infer_layout(shape):
    if len(shape) != 4:
        raise ValueError(f"unsupported output rank: {shape}")
    # most common: [1, C, 1, T] (nchw)
    if shape[0] == 1 and shape[2] == 1:
        return "nchw"
    # vpm log style: [T, 1, C, 1]
    if shape[1] == 1 and shape[3] == 1:
        return "t1c1"
    raise ValueError(f"cannot infer layout from shape: {shape}")


def detect_dtype(dat_path: Path, elems: int):
    size = dat_path.stat().st_size
    if size == elems:
        return np.uint8, "uint8"
    if size == elems * 4:
        return np.float32, "float32"
    if size == elems * 2:
        return np.float16, "float16"
    raise ValueError(f"unexpected dat size={size}, elems={elems}")


def reshape_to_tc(arr: np.ndarray, shape, layout: str) -> np.ndarray:
    if layout == "nchw":
        n, c, h, t = shape
        x = arr.reshape(n, c, h, t)[0, :, 0, :].T
        return x
    if layout == "t1c1":
        t, one1, c, one2 = shape
        x = arr.reshape(t, one1, c, one2)[:, 0, :, 0]
        return x
    raise ValueError(f"unsupported layout: {layout}")


def ctc_greedy(frame_ids: np.ndarray, blank_id: int):
    out = []
    prev = None
    for idx in frame_ids.tolist():
        if idx != blank_id and idx != prev:
            out.append(int(idx))
        prev = int(idx)
    return out


def main():
    p = argparse.ArgumentParser(description="Decode Korean Citrinet output_0.dat to text")
    p.add_argument("--dat", required=True, help="output_0.dat path")
    p.add_argument("--meta", required=True, help="nbg_meta.json path")
    p.add_argument("--model-file", required=True, help=".nemo model path for tokenizer")
    p.add_argument("--layout", choices=["auto", "nchw", "t1c1"], default="auto")
    p.add_argument("--blank-id", type=int, default=-1, help="default: use model blank id")
    p.add_argument("--dequant", action="store_true", help="dequantize uint8 logits before argmax")
    args = p.parse_args()

    dat_path = Path(args.dat)
    meta_path = Path(args.meta)

    shape, q = load_meta(meta_path)
    layout = infer_layout(shape) if args.layout == "auto" else args.layout
    elems = int(np.prod(shape))
    dtype, dtype_name = detect_dtype(dat_path, elems)

    raw = np.fromfile(dat_path, dtype=dtype)
    if raw.size != elems:
        raise ValueError(f"size mismatch: got={raw.size}, expected={elems}")
    logits = reshape_to_tc(raw, shape, layout).astype(np.float32, copy=False)

    if args.dequant and dtype_name == "uint8":
        scale = float(q.get("scale", 1.0))
        zp = int(q.get("zero_point", 0))
        logits = (logits - zp) * scale

    asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        restore_path=args.model_file, map_location="cpu"
    )
    asr.eval()

    blank_id = args.blank_id
    if blank_id < 0:
        blank_id = int(asr.decoder.num_classes_with_blank - 1)

    frame_ids = logits.argmax(axis=1).astype(np.int64)
    token_ids = ctc_greedy(frame_ids, blank_id=blank_id)
    text = asr.tokenizer.ids_to_text(token_ids).strip()

    print(f"[INFO] dat={dat_path}")
    print(f"[INFO] shape={shape}, layout={layout}, dtype={dtype_name}")
    print(f"[INFO] frames={len(frame_ids)}, blank_id={blank_id}")
    print(f"[INFO] token_ids={token_ids}")
    print(f"[TEXT] {text}")


if __name__ == "__main__":
    main()


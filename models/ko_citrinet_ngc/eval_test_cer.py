#!/usr/bin/env python3
import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import nemo.collections.asr as nemo_asr


def norm_text(s: str, mode: str = "strip_space") -> str:
    if mode == "strip_space":
        s = s.strip()
        s = re.sub(r"\s+", "", s)
        return s
    if mode == "raw":
        return s
    raise ValueError(f"unsupported norm mode: {mode}")


def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def ctc_decode_ids(argmax_ids: np.ndarray, blank_id: int):
    out = []
    prev = None
    for x in argmax_ids.tolist():
        if x != blank_id and x != prev:
            out.append(int(x))
        prev = int(x)
    return out


def load_pegasus_iter_argmax(infer_out_dir: Path, i: int, classes: int, frames: int):
    pats = [
        str(infer_out_dir / f"iter_{i}_*{classes}_1_{frames}.tensor"),
        str(infer_out_dir / f"iter_{i}_*out0*.tensor"),
    ]
    cand = []
    for p in pats:
        cand.extend(glob.glob(p))
    if not cand:
        raise FileNotFoundError(f"missing output tensor for iter={i} in {infer_out_dir}")
    p = sorted(set(cand))[0]
    vals = np.loadtxt(p, dtype=np.float32)
    expect = 1 * classes * 1 * frames
    if vals.size != expect:
        raise ValueError(f"bad tensor size {vals.size}, expect {expect}, file={p}")
    y = vals.reshape(1, classes, 1, frames)
    return y[0, :, 0, :].T.argmax(axis=1).astype(np.int64)


def main():
    p = argparse.ArgumentParser(description="Evaluate CER on Korean test set")
    p.add_argument("--mode", choices=["onnx", "pegasus"], required=True)
    p.add_argument("--meta-tsv", required=True, help="test_dataset.meta.tsv")
    p.add_argument("--model-file", required=True, help=".nemo model (for tokenizer)")
    p.add_argument("--onnx", default="", help="onnx path (mode=onnx)")
    p.add_argument("--pegasus-out-dir", default="", help="infer output dir (mode=pegasus)")
    p.add_argument("--classes", type=int, default=2049)
    p.add_argument("--frames", type=int, default=38)
    p.add_argument(
        "--norm-mode",
        choices=["strip_space", "raw"],
        default="strip_space",
        help="strip_space: remove all whitespaces before CER, raw: keep exact characters",
    )
    p.add_argument("--out-tsv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.meta_tsv, sep="\t")
    rows = df.to_dict("records")
    if not rows:
        raise RuntimeError("empty meta tsv")

    asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        restore_path=args.model_file, map_location="cpu"
    )
    asr.eval()
    blank_id = asr.decoder.num_classes_with_blank - 1

    sess = None
    if args.mode == "onnx":
        if not args.onnx:
            raise ValueError("--onnx is required for mode=onnx")
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    else:
        if not args.pegasus_out_dir:
            raise ValueError("--pegasus-out-dir is required for mode=pegasus")
        infer_out_dir = Path(args.pegasus_out_dir)

    out_rows = []
    for i, r in enumerate(rows):
        gt = str(r["text"])
        npy_path = Path(r["npy_path"])
        x = np.load(npy_path).astype(np.float32)

        if args.mode == "onnx":
            feed = {}
            input_names = [i.name for i in sess.get_inputs()]
            if "audio_signal" in input_names:
                feed["audio_signal"] = x
            else:
                feed[input_names[0]] = x

            if "audio_signal_length" in input_names:
                # x shape: [1,80,1,T]
                feed["audio_signal_length"] = np.array([x.shape[-1]], dtype=np.int64)
            elif len(input_names) >= 2:
                # fallback for non-standard 2nd input name
                feed[input_names[1]] = np.array([x.shape[-1]], dtype=np.int64)

            y = sess.run(None, feed)[0]  # usually [1,C,1,T]
            argmax_ids = y[0, :, 0, :].T.argmax(axis=1).astype(np.int64)
        else:
            argmax_ids = load_pegasus_iter_argmax(
                infer_out_dir, i, classes=args.classes, frames=args.frames
            )

        tok_ids = ctc_decode_ids(argmax_ids, blank_id=blank_id)
        pred = asr.tokenizer.ids_to_text(tok_ids)

        gt_n = norm_text(gt, mode=args.norm_mode)
        pr_n = norm_text(pred, mode=args.norm_mode)
        ed = edit_distance(gt_n, pr_n)
        cer = ed / max(1, len(gt_n))
        out_rows.append(
            {
                "idx": i,
                "wav_path": r.get("wav_path", ""),
                "gt": gt,
                "pred": pred,
                "gt_norm": gt_n,
                "pred_norm": pr_n,
                "edit_distance": ed,
                "ref_len": len(gt_n),
                "cer": cer,
            }
        )

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "wav_path",
                "gt",
                "pred",
                "gt_norm",
                "pred_norm",
                "edit_distance",
                "ref_len",
                "cer",
            ],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(out_rows)

    mean_cer = float(np.mean([r["cer"] for r in out_rows]))
    print(f"[DONE] mode={args.mode}, samples={len(out_rows)}, mean_CER={mean_cer:.4f}")
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr


def load_model(model_file: str, model_name: str):
    if model_file:
        return nemo_asr.models.EncDecCTCModelBPE.restore_from(
            restore_path=model_file, map_location="cpu"
        )
    if not model_name:
        raise ValueError("Either --model-file or --model-name is required")
    return nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)


def linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    ratio = float(dst_sr) / float(src_sr)
    out_len = max(1, int(round(len(audio) * ratio)))
    x0 = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x1 = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    return np.interp(x1, x0, audio).astype(np.float32)


def read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            wav = row.get("wav_path", "").strip()
            txt = row.get("text", "").strip()
            if wav:
                rows.append((Path(wav), txt))
    return rows


def main():
    p = argparse.ArgumentParser(description="Generate [1,80,1,T] npy dataset from wav manifest")
    p.add_argument("--manifest-tsv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-list", required=True)
    p.add_argument("--model-file", default="")
    p.add_argument("--model-name", default="")
    p.add_argument("--time-frames", type=int, default=300)
    p.add_argument("--target-sr", type=int, default=16000)
    args = p.parse_args()

    manifest = Path(args.manifest_tsv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_list = Path(args.out_list).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_list.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_file, args.model_name)
    model.eval()
    preprocessor = model.preprocessor

    rows = read_manifest(manifest)
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest}")

    npy_paths = []
    with torch.no_grad():
        for i, (wav_path, txt) in enumerate(rows):
            audio, sr = sf.read(str(wav_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = linear_resample(audio.astype(np.float32), sr, args.target_sr)

            sig = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            ln = torch.tensor([sig.shape[1]], dtype=torch.long)
            feat, _ = preprocessor(input_signal=sig, length=ln)

            t = feat.shape[2]
            if t > args.time_frames:
                feat = feat[:, :, : args.time_frames]
            elif t < args.time_frames:
                feat = torch.nn.functional.pad(feat, (0, args.time_frames - t))

            x = feat.unsqueeze(2).cpu().numpy().astype(np.float32)
            npy_path = out_dir / f"input_{i:05d}.npy"
            np.save(npy_path, x)
            npy_paths.append((npy_path, txt, wav_path))

            if (i + 1) % 20 == 0:
                print(f"[PROGRESS] {i + 1}/{len(rows)}")

    with out_list.open("w", encoding="utf-8") as f:
        for pth, _, _ in npy_paths:
            f.write(str(pth) + "\n")

    info_tsv = out_list.with_suffix(".meta.tsv")
    with info_tsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["npy_path", "wav_path", "text"], delimiter="\t")
        w.writeheader()
        for npy_path, txt, wav_path in npy_paths:
            w.writerow({"npy_path": str(npy_path), "wav_path": str(wav_path), "text": txt})

    print("[DONE] npy dataset generated")
    print(f"manifest={manifest}")
    print(f"count={len(npy_paths)}")
    print(f"out_list={out_list}")
    print(f"meta={info_tsv}")


if __name__ == "__main__":
    main()

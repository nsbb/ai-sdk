#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr


def linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    ratio = float(dst_sr) / float(src_sr)
    out_len = max(1, int(round(len(audio) * ratio)))
    x0 = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x1 = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    return np.interp(x1, x0, audio).astype(np.float32)


def to_qparams(qtype: str):
    q = qtype.lower()
    if q in ("u8", "uint8"):
        return 0, 255, np.uint8
    if q in ("i8", "int8"):
        return -128, 127, np.int8
    raise ValueError(f"unsupported qtype: {qtype}")


def main():
    p = argparse.ArgumentParser(
        description="Create real 5s wav and corresponding model input npy/dat for Citrinet NB"
    )
    p.add_argument("--wav", required=True, help="source wav path")
    p.add_argument("--model-file", required=True, help=".nemo model path")
    p.add_argument("--meta", required=True, help="nbg_meta.json (for quantized dat)")
    p.add_argument("--out-wav", required=True, help="output 5s wav path")
    p.add_argument("--out-float-npy", required=True, help="output float npy path")
    p.add_argument("--out-dat", required=True, help="output quantized input dat path")
    p.add_argument("--out-fp32-dat", default="", help="optional output fp32 dat path")
    p.add_argument("--duration-sec", type=float, default=5.0)
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--time-frames", type=int, default=500)
    args = p.parse_args()

    wav_path = Path(args.wav).resolve()
    model_path = Path(args.model_file).resolve()
    meta_path = Path(args.meta).resolve()

    out_wav = Path(args.out_wav).resolve()
    out_npy = Path(args.out_float_npy).resolve()
    out_dat = Path(args.out_dat).resolve()
    out_fp32_dat = Path(args.out_fp32_dat).resolve() if args.out_fp32_dat else None

    if not wav_path.is_file():
        raise FileNotFoundError(f"wav not found: {wav_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta not found: {meta_path}")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_dat.parent.mkdir(parents=True, exist_ok=True)
    if out_fp32_dat is not None:
        out_fp32_dat.parent.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = linear_resample(audio.astype(np.float32), sr, args.target_sr)

    target_samples = int(round(args.duration_sec * args.target_sr))
    if len(audio) > target_samples:
        audio_5s = audio[:target_samples]
    elif len(audio) < target_samples:
        audio_5s = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
    else:
        audio_5s = audio

    sf.write(str(out_wav), audio_5s, args.target_sr)

    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        restore_path=str(model_path),
        map_location="cpu",
    )
    model.eval()
    preprocessor = model.preprocessor

    with torch.no_grad():
        sig = torch.tensor(audio_5s, dtype=torch.float32).unsqueeze(0)
        ln = torch.tensor([sig.shape[1]], dtype=torch.long)
        feat, _ = preprocessor(input_signal=sig, length=ln)

        t = feat.shape[2]
        if t > args.time_frames:
            feat = feat[:, :, : args.time_frames]
        elif t < args.time_frames:
            feat = torch.nn.functional.pad(feat, (0, args.time_frames - t))

        x = feat.unsqueeze(2).cpu().numpy().astype(np.float32)

    np.save(out_npy, x)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    first_input = next(iter(meta["Inputs"].values()))
    expected_shape = first_input["shape"]
    q = first_input["quantize"]

    if list(x.shape) != list(expected_shape):
        raise ValueError(
            f"shape mismatch: got {list(x.shape)}, expected {expected_shape}. "
            "Adjust --time-frames or use matching NB/meta."
        )

    qtype = q["qtype"]
    scale = float(q["scale"])
    zero_point = int(q["zero_point"])
    if scale <= 0:
        raise ValueError(f"invalid scale in meta: {scale}")

    qmin, qmax, qdtype = to_qparams(qtype)
    y = np.round(x / scale + zero_point)
    y = np.clip(y, qmin, qmax).astype(qdtype)
    y.tofile(out_dat)

    if out_fp32_dat is not None:
        x.astype(np.float32, copy=False).tofile(out_fp32_dat)

    print("[DONE] 5s input created")
    print(f"source_wav={wav_path}")
    print(f"wav_5s={out_wav}")
    print(f"float_npy={out_npy}, shape={list(x.shape)}")
    print(f"quant_dat={out_dat}, qtype={qtype}, scale={scale}, zero_point={zero_point}")
    if out_fp32_dat is not None:
        print(f"fp32_dat={out_fp32_dat}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zeroth-Korean arrow → wav2vec2 calibration npy 생성.

Zeroth-Korean test split (457개) + train split에서 audio를 읽어
3초 (48000 samples, 16kHz) 길이로 pad/truncate 후 [1, 48000] float32 npy로 저장.

Usage:
    python3 make_zeroth_calib.py --output-dir calib_npy --count 100
    python3 make_zeroth_calib.py --output-dir calib_npy --count 100 --split test
"""
import argparse
import io
import numpy as np
import pyarrow as pa
import soundfile as sf
from pathlib import Path

ARROW_DIR = Path(__file__).resolve().parent.parent / "wav2vec2_ko_base_3s/zeroth_korean_cache/kresnik___zeroth_korean/default/0.0.0/1fe937899f828af822293d05e086200946088bdf"
SAMPLE_RATE = 16000
TARGET_LEN = 48000  # 3 seconds


def load_arrow_table(split="test"):
    if split == "test":
        arrow_file = ARROW_DIR / "zeroth_korean-test.arrow"
        with open(arrow_file, "rb") as f:
            return pa.ipc.open_stream(f).read_all()
    else:
        tables = []
        for i in range(6):
            arrow_file = ARROW_DIR / f"zeroth_korean-train-{i:05d}-of-00006.arrow"
            if arrow_file.exists():
                with open(arrow_file, "rb") as f:
                    tables.append(pa.ipc.open_stream(f).read_all())
        return pa.concat_tables(tables)


def decode_audio(audio_bytes):
    """FLAC bytes → float32 numpy array."""
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data.astype(np.float32)


def pad_or_truncate(audio, target_len=TARGET_LEN):
    """Pad with zeros or truncate to exact target length."""
    if len(audio) >= target_len:
        return audio[:target_len]
    else:
        return np.pad(audio, (0, target_len - len(audio)), mode='constant')


def wav2vec2_normalize(audio):
    """Apply Wav2Vec2FeatureExtractor normalization (zero-mean, unit-variance)."""
    mean = audio.mean()
    var = audio.var()
    if var > 0:
        audio = (audio - mean) / np.sqrt(var + 1e-7)
    else:
        audio = audio - mean
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="calib_npy", help="Output directory for npy files")
    parser.add_argument("--count", type=int, default=100, help="Number of calibration samples")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Minimum audio duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Zeroth-Korean {args.split} split...")
    table = load_arrow_table(args.split)
    n_total = len(table)
    print(f"  Total samples: {n_total}")

    # Shuffle indices
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_total)

    saved = 0
    skipped_short = 0
    texts = []

    for idx in indices:
        if saved >= args.count:
            break

        row = {c: table.column(c)[int(idx)].as_py() for c in table.column_names}
        audio_bytes = row['audio']['bytes']
        text = row['text']

        audio = decode_audio(audio_bytes)
        duration = len(audio) / SAMPLE_RATE

        if duration < args.min_duration:
            skipped_short += 1
            continue

        # Pad or truncate
        audio = pad_or_truncate(audio)

        # NOTE: Do NOT normalize here — wav2vec2 has internal GroupNorm
        # in the feature extractor. Unnormalized audio gives better uint8
        # quantization precision (narrower range → finer scale).

        # Save as [1, 48000] float32
        npy_data = audio.reshape(1, -1)
        npy_path = out_dir / f"calib_{saved:04d}.npy"
        np.save(npy_path, npy_data)

        texts.append(f"calib_{saved:04d}\t{text}\t{duration:.2f}s")
        saved += 1

    # Save manifest
    manifest_path = out_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# Zeroth-Korean calibration data\n")
        f.write(f"# Split: {args.split}, Count: {saved}, Seed: {args.seed}\n")
        f.write(f"# Min duration: {args.min_duration}s, Target: {TARGET_LEN} samples ({TARGET_LEN/SAMPLE_RATE}s)\n")
        f.write("# ID\tText\tOriginal Duration\n")
        for t in texts:
            f.write(t + "\n")

    print(f"\nDone: {saved} files saved to {out_dir}/")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Manifest: {manifest_path}")

    # Verify first file
    verify = np.load(out_dir / "calib_0000.npy")
    print(f"\nVerification - calib_0000.npy:")
    print(f"  Shape: {verify.shape}, dtype: {verify.dtype}")
    print(f"  Range: [{verify.min():.6f}, {verify.max():.6f}]")


if __name__ == "__main__":
    main()

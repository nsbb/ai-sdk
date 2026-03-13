import os
import argparse
import numpy as np

def make_waveform(n: int, sr: int, seed: int) -> np.ndarray:
    """
    Create a pseudo-speech-like waveform (float32) in [-1, 1].
    Mix of sine bursts + noise + random amplitude envelope.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / np.float32(sr)

    # Random envelope (piecewise linear)
    num_knots = 8
    knot_x = np.linspace(0, n - 1, num_knots, dtype=np.int32)
    knot_y = rng.uniform(0.1, 1.0, size=num_knots).astype(np.float32)
    env = np.interp(np.arange(n), knot_x, knot_y).astype(np.float32)

    # A few sine components (formant-ish)
    freqs = rng.choice([120, 180, 220, 300, 440, 660, 880, 1200], size=3, replace=False)
    phases = rng.uniform(0, 2*np.pi, size=3).astype(np.float32)
    sines = sum(np.sin(2*np.pi*float(f)*t + float(p)).astype(np.float32) for f, p in zip(freqs, phases))
    sines *= (0.25 / 3.0)

    # Noise (low amplitude)
    noise = rng.normal(0.0, 0.03, size=n).astype(np.float32)

    # Occasional burst (simulate consonant energy)
    burst = np.zeros(n, dtype=np.float32)
    for _ in range(rng.integers(2, 6)):
        center = int(rng.integers(int(0.1*n), int(0.9*n)))
        width = int(rng.integers(int(0.005*n), int(0.03*n)))
        start = max(0, center - width)
        end = min(n, center + width)
        burst[start:end] += rng.normal(0.0, 0.08, size=(end-start)).astype(np.float32)

    x = (sines + noise + burst) * env

    # Normalize to ~ -12 dBFS-ish
    peak = np.max(np.abs(x)) + 1e-8
    x = x / peak * np.float32(0.25)

    # Clip to [-1, 1]
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="dummy_dataset", help="output directory")
    ap.add_argument("--num", type=int, default=20, help="number of samples")
    ap.add_argument("--seconds", type=float, default=2.0, help="duration seconds")
    ap.add_argument("--sr", type=int, default=16000, help="sample rate")
    ap.add_argument("--prefix", default="wav2vec2_in", help="filename prefix")
    ap.add_argument("--seed", type=int, default=777, help="random seed base")
    ap.add_argument("--make_dataset_txt", action="store_true", help="write dataset.txt listing npy files")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n = int(args.seconds * args.sr)

    paths = []
    for i in range(args.num):
        x = make_waveform(n=n, sr=args.sr, seed=args.seed + i)

        # ACUITY input often expects [1, T]
        x2 = x.reshape(1, n).astype(np.float32)

        fn = f"{args.prefix}_{args.seconds:.3f}s_{i:03d}.npy".replace(".", "p")
        p = os.path.join(args.outdir, fn)
        np.save(p, x2)
        paths.append(p)

    print(f"[OK] Wrote {len(paths)} npy files to: {args.outdir}")
    print(f"[INFO] Each shape: (1, {n}), dtype=float32")

    if args.make_dataset_txt:
        dataset_path = os.path.join(args.outdir, "dataset.txt")
        with open(dataset_path, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(p + "\n")
        print(f"[OK] Wrote dataset list: {dataset_path}")

if __name__ == "__main__":
    main()
import os
import argparse
import numpy as np

def make_waveform(n: int, sr: int, seed: int) -> np.ndarray:
    """
    Create a pseudo-speech-like waveform (float32) in [-1, 1].
    Mix of sine bursts + noise + random amplitude envelope.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / np.float32(sr)

    # Random envelope (piecewise linear)
    num_knots = 8
    knot_x = np.linspace(0, n - 1, num_knots, dtype=np.int32)
    knot_y = rng.uniform(0.1, 1.0, size=num_knots).astype(np.float32)
    env = np.interp(np.arange(n), knot_x, knot_y).astype(np.float32)

    # A few sine components (formant-ish)
    freqs = rng.choice([120, 180, 220, 300, 440, 660, 880, 1200], size=3, replace=False)
    phases = rng.uniform(0, 2*np.pi, size=3).astype(np.float32)
    sines = sum(np.sin(2*np.pi*float(f)*t + float(p)).astype(np.float32) for f, p in zip(freqs, phases))
    sines *= (0.25 / 3.0)

    # Noise (low amplitude)
    noise = rng.normal(0.0, 0.03, size=n).astype(np.float32)

    # Occasional burst (simulate consonant energy)
    burst = np.zeros(n, dtype=np.float32)
    for _ in range(rng.integers(2, 6)):
        center = int(rng.integers(int(0.1*n), int(0.9*n)))
        width = int(rng.integers(int(0.005*n), int(0.03*n)))
        start = max(0, center - width)
        end = min(n, center + width)
        burst[start:end] += rng.normal(0.0, 0.08, size=(end-start)).astype(np.float32)

    x = (sines + noise + burst) * env

    # Normalize to ~ -12 dBFS-ish
    peak = np.max(np.abs(x)) + 1e-8
    x = x / peak * np.float32(0.25)

    # Clip to [-1, 1]
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="dummy_dataset", help="output directory")
    ap.add_argument("--num", type=int, default=20, help="number of samples")
    ap.add_argument("--seconds", type=float, default=2.0, help="duration seconds")
    ap.add_argument("--sr", type=int, default=16000, help="sample rate")
    ap.add_argument("--prefix", default="wav2vec2_in", help="filename prefix")
    ap.add_argument("--seed", type=int, default=777, help="random seed base")
    ap.add_argument("--make_dataset_txt", action="store_true", help="write dataset.txt listing npy files")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n = int(args.seconds * args.sr)

    paths = []
    for i in range(args.num):
        x = make_waveform(n=n, sr=args.sr, seed=args.seed + i)

        # ACUITY input often expects [1, T]
        x2 = x.reshape(1, n).astype(np.float32)

        fn = f"{args.prefix}_{args.seconds:.3f}s_{i:03d}.npy".replace(".", "p")
        p = os.path.join(args.outdir, fn)
        np.save(p, x2)
        paths.append(p)

    print(f"[OK] Wrote {len(paths)} npy files to: {args.outdir}")
    print(f"[INFO] Each shape: (1, {n}), dtype=float32")

    if args.make_dataset_txt:
        dataset_path = os.path.join(args.outdir, "dataset.txt")
        with open(dataset_path, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(p + "\n")
        print(f"[OK] Wrote dataset list: {dataset_path}")

if __name__ == "__main__":
    main()


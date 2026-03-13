#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

def collect_and_copy(
    csv_path: Path,
    wav_col: str,
    text_col: str,
    max_rows: int,
    calib_count: int,
    test_count: int,
    out_dir: Path,
):
    out_calib_dir = out_dir / "wav_calib"
    out_test_dir = out_dir / "wav_test"
    out_calib_dir.mkdir(parents=True, exist_ok=True)
    out_test_dir.mkdir(parents=True, exist_ok=True)

    calib_out = []
    test_out = []
    seen = set()
    processed = 0
    copied = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"empty csv: {csv_path}")
        if wav_col not in reader.fieldnames:
            raise KeyError(f"missing wav column: {wav_col}")
        if text_col not in reader.fieldnames:
            raise KeyError(f"missing text column: {text_col}")

        for r in reader:
            if max_rows > 0 and processed >= max_rows:
                break
            if len(calib_out) >= calib_count and len(test_out) >= test_count:
                break

            processed += 1
            wav = str(r.get(wav_col, "")).strip()
            if not wav or wav in seen:
                continue
            lw = wav.lower()
            if not (lw.endswith(".wav") or lw.endswith(".flac")):
                continue
            seen.add(wav)

            src = Path(wav)
            txt = str(r.get(text_col, "")).strip()

            if len(calib_out) < calib_count:
                idx = len(calib_out)
                dst = out_calib_dir / f"calib_{idx:05d}{src.suffix.lower() or '.wav'}"
            else:
                idx = len(test_out)
                dst = out_test_dir / f"test_{idx:05d}{src.suffix.lower() or '.wav'}"

            try:
                shutil.copy2(src, dst)
            except Exception:
                continue

            copied += 1
            row = (dst.resolve(), txt, src.resolve())
            if len(calib_out) < calib_count:
                calib_out.append(row)
            else:
                test_out.append(row)

            if copied % 20 == 0:
                print(f"[PROGRESS] copied={copied}, calib={len(calib_out)}, test={len(test_out)}")

    return calib_out, test_out


def write_manifest_tsv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["wav_path", "text", "src_path"], delimiter="\t")
        w.writeheader()
        for wav, txt, src in rows:
            w.writerow({"wav_path": str(wav), "text": txt, "src_path": str(src)})


def write_wav_list(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for wav, _, _ in rows:
            f.write(str(wav) + "\n")


def main():
    p = argparse.ArgumentParser(description="Copy Korean wav subset from CSV manifest")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--wav-col", default="raw_data")
    p.add_argument("--text-col", default="transcript")
    p.add_argument("--calib-count", type=int, default=120)
    p.add_argument("--test-count", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Read only first N CSV rows for faster debug (0 means all rows)",
    )
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    csv_path = Path(args.csv_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    calib_out, test_out = collect_and_copy(
        csv_path=csv_path,
        wav_col=args.wav_col,
        text_col=args.text_col,
        max_rows=args.max_rows,
        calib_count=args.calib_count,
        test_count=args.test_count,
        out_dir=out_dir,
    )

    if len(calib_out) < args.calib_count or len(test_out) < args.test_count:
        raise RuntimeError(
            f"copied fewer files than requested: calib {len(calib_out)}/{args.calib_count}, "
            f"test {len(test_out)}/{args.test_count}. increase --max-rows."
        )

    write_manifest_tsv(calib_out, out_dir / "calib_manifest.tsv")
    write_manifest_tsv(test_out, out_dir / "test_manifest.tsv")
    write_wav_list(calib_out, out_dir / "calib_wavs.txt")
    write_wav_list(test_out, out_dir / "test_wavs.txt")

    first_wav, first_txt, _ = test_out[0]
    with (out_dir / "test_first.txt").open("w", encoding="utf-8") as f:
        f.write(f"wav_path\t{first_wav}\n")
        f.write(f"text\t{first_txt}\n")

    print("[DONE] copied wav subset")
    print(f"csv={csv_path}")
    print(f"calib={len(calib_out)}, test={len(test_out)}")
    print(f"out_dir={out_dir}")
    print(f"first_test_wav={first_wav}")
    print(f"first_test_text={first_txt}")


if __name__ == "__main__":
    main()

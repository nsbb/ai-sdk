#!/usr/bin/env python3
"""Prepare Korean WAV files as uint8 input_0.dat for wav2vec2-base-korean NPU model.

Usage:
    python3 prepare_ko_test_input.py <wav_path> [--output_dir <dir>] [--gt "정답 텍스트"]
    python3 prepare_ko_test_input.py --batch  # Process all known Korean test WAVs

Model spec (from nbg_meta.json):
    Input:  [1, 48000] float32 -> uint8, scale=0.06662226468324661, zp=131
    Output: [1, 149, 56] uint8
    Preprocessing: Wav2Vec2FeatureExtractor (normalize=True, sr=16000)
"""
import argparse
import json
import os
import sys
import wave

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SR = 16000
TARGET_SAMPLES = 48000  # 3 seconds @ 16kHz

# Quantization parameters from nbg_meta.json
INPUT_SCALE = 0.06662226468324661
INPUT_ZP = 131

# ── Known Korean test WAV sources ────────────────────────────────────────
# ko_citrinet test WAVs with ground truth
KO_CITRINET_TEST_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "ko_citrinet_ngc", "data", "wav_test"
)
KO_CITRINET_GT = {
    "test_00001.wav": "열선 시트 조금만 뒤로 옮겨",
    "test_00002.wav": "어느 방에 에어컨 작동 중이야",
    "test_00003.wav": "그 내용이 아직은 정확하지 않아요 고객님",
    "test_00004.wav": "극도의 공포와 불안을 느끼는 것 같아요",
    "test_00005.wav": "누가 더 빨리 올라가나 내기하지",
    "test_00006.wav": "내가 좋아하는 음식 같이 먹는 게 더 좋아요",
    "test_00007.wav": "줄 서면 하나씩 줄게",
    "test_00008.wav": "그니까 수거를 했다고 지금 그러니까",
    "test_00009.wav": "그런데 관용사가 오지 않았어요",
}

# Zipformer Korean test WAVs with ground truth
ZIPFORMER_TEST_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "zipformer",
    "sherpa-onnx-streaming-zipformer-korean-2024-06-16", "test_wavs"
)
ZIPFORMER_GT = {
    "0.wav": "그는 괜찮은 척하려고 애쓰는 것 같았다",
    "1.wav": "지하철에서 다리를 벌리고 앉지 마라",
    "2.wav": "부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다",
    "3.wav": "주민등록증을 보여 주시겠어요",
}


def load_and_resample_wav(wav_path: str) -> np.ndarray:
    """Load WAV file and resample to 16kHz mono float32 in [-1, 1]."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        print(f"  Resampled {sr}Hz -> {TARGET_SR}Hz")

    return audio


def wav2vec2_normalize(audio: np.ndarray) -> np.ndarray:
    """Apply Wav2Vec2FeatureExtractor normalization (zero-mean, unit-variance)."""
    mean = audio.mean()
    var = audio.var()
    if var > 0:
        audio = (audio - mean) / np.sqrt(var + 1e-7)
    else:
        audio = audio - mean
    return audio


def truncate_pad(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Truncate or zero-pad audio to target_len samples."""
    if len(audio) >= target_len:
        audio = audio[:target_len]
    else:
        pad_len = target_len - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant", constant_values=0.0)
    return audio


def quantize_to_uint8(audio_fp32: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """Quantize float32 to uint8: q = clamp(round(x / scale) + zp, 0, 255)."""
    q = np.round(audio_fp32 / scale) + zp
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q


def process_wav(wav_path: str, output_dir: str, gt_text: str = None) -> dict:
    """Process a single WAV file into input_0.dat + metadata.

    Returns dict with processing info.
    """
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing: {wav_path}")
    if gt_text:
        print(f"  GT: {gt_text}")

    # 1. Load and resample
    audio = load_and_resample_wav(wav_path)
    orig_len = len(audio)
    orig_dur = orig_len / TARGET_SR
    print(f"  Loaded: {orig_len} samples ({orig_dur:.2f}s)")

    # 2. Truncate/pad to 3 seconds
    audio = truncate_pad(audio, TARGET_SAMPLES)
    if orig_len > TARGET_SAMPLES:
        print(f"  Truncated: {orig_len} -> {TARGET_SAMPLES} samples (cut {(orig_len - TARGET_SAMPLES)/TARGET_SR:.2f}s)")
    elif orig_len < TARGET_SAMPLES:
        print(f"  Padded: {orig_len} -> {TARGET_SAMPLES} samples (added {(TARGET_SAMPLES - orig_len)/TARGET_SR:.2f}s silence)")

    # 3. Normalize (Wav2Vec2FeatureExtractor: zero-mean, unit-variance)
    audio_norm = wav2vec2_normalize(audio)
    print(f"  Normalized: mean={audio_norm.mean():.6f}, std={audio_norm.std():.6f}, "
          f"range=[{audio_norm.min():.4f}, {audio_norm.max():.4f}]")

    # 4. Save float32 numpy
    audio_fp32 = audio_norm.reshape(1, TARGET_SAMPLES).astype(np.float32)
    npy_path = os.path.join(output_dir, f"{basename}.npy")
    np.save(npy_path, audio_fp32)
    print(f"  Saved npy: {npy_path} (shape={audio_fp32.shape}, dtype={audio_fp32.dtype})")

    # 5. Quantize to uint8
    audio_u8 = quantize_to_uint8(audio_fp32.flatten(), INPUT_SCALE, INPUT_ZP)
    dat_path = os.path.join(output_dir, f"input_0.dat")
    audio_u8.tofile(dat_path)
    print(f"  Saved dat: {dat_path} ({len(audio_u8)} bytes)")

    # Also save per-file dat for batch testing
    per_file_dat = os.path.join(output_dir, f"{basename}_input_0.dat")
    audio_u8.tofile(per_file_dat)

    # 6. Verification
    # Dequantize back and compare
    dequant = (audio_u8.astype(np.float32) - INPUT_ZP) * INPUT_SCALE
    quant_error = np.abs(audio_fp32.flatten() - dequant)
    print(f"  Quantization error: max={quant_error.max():.6f}, mean={quant_error.mean():.6f}")

    info = {
        "wav_path": wav_path,
        "basename": basename,
        "npy_path": npy_path,
        "dat_path": dat_path,
        "original_samples": orig_len,
        "original_duration_s": round(orig_dur, 2),
        "target_samples": TARGET_SAMPLES,
        "fp32_range": [float(audio_norm.min()), float(audio_norm.max())],
        "uint8_range": [int(audio_u8.min()), int(audio_u8.max())],
        "quant_max_error": float(quant_error.max()),
        "gt_text": gt_text,
    }
    return info


def run_batch(output_base_dir: str):
    """Process all known Korean test WAVs."""
    results = []

    # 1. ko_citrinet test WAVs
    if os.path.isdir(KO_CITRINET_TEST_DIR):
        print("=" * 60)
        print(f"ko_citrinet test WAVs: {KO_CITRINET_TEST_DIR}")
        print("=" * 60)
        for fname, gt in sorted(KO_CITRINET_GT.items()):
            wav_path = os.path.join(KO_CITRINET_TEST_DIR, fname)
            if os.path.isfile(wav_path):
                out_dir = os.path.join(output_base_dir, "ko_citrinet")
                info = process_wav(wav_path, out_dir, gt_text=gt)
                results.append(info)
            else:
                print(f"  SKIP (not found): {wav_path}")
    else:
        print(f"ko_citrinet test dir not found: {KO_CITRINET_TEST_DIR}")

    # 2. Zipformer Korean test WAVs
    if os.path.isdir(ZIPFORMER_TEST_DIR):
        print("\n" + "=" * 60)
        print(f"Zipformer Korean test WAVs: {ZIPFORMER_TEST_DIR}")
        print("=" * 60)
        for fname, gt in sorted(ZIPFORMER_GT.items()):
            wav_path = os.path.join(ZIPFORMER_TEST_DIR, fname)
            if os.path.isfile(wav_path):
                out_dir = os.path.join(output_base_dir, "zipformer")
                info = process_wav(wav_path, out_dir, gt_text=gt)
                results.append(info)
            else:
                print(f"  SKIP (not found): {wav_path}")
    else:
        print(f"Zipformer test dir not found: {ZIPFORMER_TEST_DIR}")

    # Save manifest
    manifest_path = os.path.join(output_base_dir, "test_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Manifest saved: {manifest_path}")
    print(f"Total files processed: {len(results)}")

    # Create sample.txt for vpm_run (pointing to last processed file)
    if results:
        sample_txt_path = os.path.join(output_base_dir, "sample.txt")
        # Write vpm_run sample.txt format
        with open(sample_txt_path, "w") as f:
            # Use the NB from the parent dir's wksp
            f.write("[network]\n")
            f.write("network_binary.nb\n")
            f.write("[input]\n")
            f.write("input_0.dat\n")
            f.write("[output]\n")
            f.write("output_0.dat\n")
        print(f"sample.txt: {sample_txt_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Korean WAV -> uint8 input for wav2vec2-base-korean NPU"
    )
    parser.add_argument("wav_path", nargs="?", help="Path to a single WAV file")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Output directory (default: <script_dir>/test_data/)")
    parser.add_argument("--gt", default=None, help="Ground truth text for the WAV")
    parser.add_argument("--batch", action="store_true",
                        help="Process all known Korean test WAV files")
    args = parser.parse_args()

    default_output = os.path.join(SCRIPT_DIR, "test_data")

    if args.batch:
        output_dir = args.output_dir or default_output
        run_batch(output_dir)
    elif args.wav_path:
        output_dir = args.output_dir or default_output
        info = process_wav(args.wav_path, output_dir, gt_text=args.gt)
        print(f"\nDone. Files saved to: {output_dir}")
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 prepare_ko_test_input.py /path/to/test.wav --gt '안녕하세요'")
        print("  python3 prepare_ko_test_input.py --batch")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare English test audio from LibriSpeech test-clean for Wav2Vec2 CER evaluation.
Extracts 50 samples, converts to 16kHz mono WAV (5 seconds), creates ground truth file.
"""
import os
import sys
import glob
import random
import struct
import numpy as np

LIBRISPEECH_DIR = "data/librispeech_test/LibriSpeech/test-clean"
OUTPUT_DIR = "data/english_test"
TARGET_SR = 16000
TARGET_LENGTH = 80000  # 5 seconds at 16kHz
NUM_SAMPLES = 50

def read_flac(flac_path):
    """Read FLAC file using soundfile."""
    import soundfile as sf
    data, sr = sf.read(flac_path, dtype='float32')
    # LibriSpeech is already 16kHz mono, but verify
    if sr != TARGET_SR:
        raise RuntimeError(f"Unexpected sample rate {sr} (expected {TARGET_SR})")
    return data

def write_wav(filepath, data, sr=16000):
    """Write float32 audio data as 16-bit PCM WAV."""
    pcm16 = (data * 32767).astype(np.int16)
    num_samples = len(pcm16)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample

    with open(filepath, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # PCM format
        f.write(struct.pack('<H', 1))   # mono
        f.write(struct.pack('<I', sr))  # sample rate
        f.write(struct.pack('<I', sr * 2))  # byte rate
        f.write(struct.pack('<H', 2))   # block align
        f.write(struct.pack('<H', 16))  # bits per sample
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm16.tobytes())

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(LIBRISPEECH_DIR):
        print(f"ERROR: LibriSpeech directory not found: {LIBRISPEECH_DIR}")
        print("Run: tar xzf test-clean.tar.gz -C data/librispeech_test/")
        sys.exit(1)

    # Collect all FLAC files and their transcripts
    trans_files = glob.glob(os.path.join(LIBRISPEECH_DIR, "*/*/*.trans.txt"))
    print(f"Found {len(trans_files)} transcript files")

    all_samples = []
    for trans_file in trans_files:
        trans_dir = os.path.dirname(trans_file)
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    flac_path = os.path.join(trans_dir, utt_id + ".flac")
                    if os.path.exists(flac_path):
                        all_samples.append((utt_id, flac_path, text))

    print(f"Total utterances: {len(all_samples)}")

    # Select samples: prefer utterances that fit in 5 seconds
    # First pass: check durations
    random.seed(42)
    random.shuffle(all_samples)

    selected = []
    skipped_long = 0

    for utt_id, flac_path, text in all_samples:
        if len(selected) >= NUM_SAMPLES:
            break

        try:
            audio = read_flac(flac_path)
            duration = len(audio) / TARGET_SR

            # Skip very short (<1s) or very long (>8s) utterances
            if duration < 1.0:
                continue
            if duration > 8.0:
                skipped_long += 1
                continue

            selected.append((utt_id, flac_path, text, audio, duration))
            print(f"  [{len(selected):2d}/{NUM_SAMPLES}] {utt_id}: {duration:.1f}s, \"{text[:60]}...\"" if len(text) > 60 else f"  [{len(selected):2d}/{NUM_SAMPLES}] {utt_id}: {duration:.1f}s, \"{text}\"")

        except Exception as e:
            print(f"  SKIP {utt_id}: {e}")
            continue

    print(f"\nSelected {len(selected)} samples (skipped {skipped_long} too-long)")

    # Write output files
    ground_truth = []

    for i, (utt_id, flac_path, text, audio, duration) in enumerate(selected):
        # Pad or truncate to TARGET_LENGTH
        if len(audio) >= TARGET_LENGTH:
            audio_5s = audio[:TARGET_LENGTH]
        else:
            audio_5s = np.pad(audio, (0, TARGET_LENGTH - len(audio)), mode='constant')

        # Save WAV
        wav_name = f"en_test_{i:04d}.wav"
        wav_path = os.path.join(OUTPUT_DIR, wav_name)
        write_wav(wav_path, audio_5s, TARGET_SR)

        # Save NPY (for vpm_run)
        npy_name = f"en_test_{i:04d}.npy"
        npy_path = os.path.join(OUTPUT_DIR, npy_name)
        np.save(npy_path, audio_5s.reshape(1, -1))

        ground_truth.append(f"{wav_name}\t{text}\t{duration:.2f}")

    # Write ground truth file
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.txt")
    with open(gt_path, 'w') as f:
        f.write("# filename\tground_truth\tduration_sec\n")
        for line in ground_truth:
            f.write(line + "\n")

    print(f"\nOutput: {OUTPUT_DIR}/")
    print(f"  {len(selected)} WAV files (en_test_XXXX.wav)")
    print(f"  {len(selected)} NPY files (en_test_XXXX.npy)")
    print(f"  ground_truth.txt")

    # Summary statistics
    durations = [s[4] for s in selected]
    print(f"\nDuration stats: min={min(durations):.1f}s, max={max(durations):.1f}s, mean={np.mean(durations):.1f}s")

if __name__ == "__main__":
    main()

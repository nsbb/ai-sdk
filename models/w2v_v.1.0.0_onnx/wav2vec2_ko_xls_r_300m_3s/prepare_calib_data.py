#!/usr/bin/env python3
"""Prepare calibration data for Korean Wav2Vec2 XLS-R-300M quantization.
Uses Korean speech from available WAV files or generates synthetic samples.
"""
import numpy as np
import os
import glob

OUTPUT_DIR = "calib_data"
INPUT_LENGTH = 48000  # 3 seconds @ 16kHz
NUM_SAMPLES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to find Korean WAV files from KoCitrinet test data
ko_wav_dirs = [
    "/home/nsbb/travail/claude/T527/ai-sdk/models/ko_citrinet_ngc/data",
    "/home/nsbb/travail/claude/T527/ai-sdk/models/ko_citrinet_ngc/v2_nb_fixlen/test_wavs",
    "/home/nsbb/travail/claude/T527/ai-sdk/models/ko_citrinet_ngc/calib_wavs",
]

wav_files = []
for d in ko_wav_dirs:
    if os.path.exists(d):
        wav_files.extend(glob.glob(os.path.join(d, "*.wav")))
        wav_files.extend(glob.glob(os.path.join(d, "**/*.wav"), recursive=True))

# Also check for npy files from English wav2vec2
en_npy_dir = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_base_960h_5s/data/english_test"
en_npy_files = []
if os.path.exists(en_npy_dir):
    en_npy_files = sorted(glob.glob(os.path.join(en_npy_dir, "*.npy")))

print(f"Found {len(wav_files)} Korean WAV files")
print(f"Found {len(en_npy_files)} English NPY files")

dataset_txt_lines = []
count = 0

# Method 1: Load WAV files
if wav_files:
    try:
        import soundfile as sf
        for wav_path in wav_files[:NUM_SAMPLES]:
            audio, sr = sf.read(wav_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            if sr != 16000:
                # Simple resample
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)
            # Normalize (wav2vec2 expects normalized input)
            if np.abs(audio).max() > 0:
                audio = (audio - audio.mean()) / (np.std(audio) + 1e-7)
            # Pad or truncate to INPUT_LENGTH
            if len(audio) < INPUT_LENGTH:
                audio = np.pad(audio, (0, INPUT_LENGTH - len(audio)))
            else:
                audio = audio[:INPUT_LENGTH]
            audio = audio.reshape(1, INPUT_LENGTH)

            npy_path = os.path.join(OUTPUT_DIR, f"calib_{count:04d}.npy")
            np.save(npy_path, audio)
            dataset_txt_lines.append(os.path.abspath(npy_path))
            count += 1
            if count >= NUM_SAMPLES:
                break
    except ImportError:
        print("soundfile not available, skipping WAV loading")

# Method 2: Use English NPY files (cross-language calibration is OK for quantization)
if count < NUM_SAMPLES and en_npy_files:
    print(f"Supplementing with English NPY files ({NUM_SAMPLES - count} more needed)")
    for npy_path in en_npy_files:
        if count >= NUM_SAMPLES:
            break
        audio = np.load(npy_path).astype(np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        # Truncate/pad to 3s
        if audio.shape[1] < INPUT_LENGTH:
            audio = np.pad(audio, ((0, 0), (0, INPUT_LENGTH - audio.shape[1])))
        else:
            audio = audio[:, :INPUT_LENGTH]

        out_path = os.path.join(OUTPUT_DIR, f"calib_{count:04d}.npy")
        np.save(out_path, audio)
        dataset_txt_lines.append(os.path.abspath(out_path))
        count += 1

# Method 3: Generate varied synthetic data if still not enough
while count < NUM_SAMPLES:
    # Random audio-like signal with varying characteristics
    np.random.seed(count)
    t = np.linspace(0, 3, INPUT_LENGTH)
    freq = np.random.uniform(100, 4000)
    audio = np.sin(2 * np.pi * freq * t) * np.random.uniform(0.1, 0.5)
    audio += np.random.randn(INPUT_LENGTH) * 0.02
    audio = (audio - audio.mean()) / (np.std(audio) + 1e-7)
    audio = audio.astype(np.float32).reshape(1, INPUT_LENGTH)

    out_path = os.path.join(OUTPUT_DIR, f"calib_{count:04d}.npy")
    np.save(out_path, audio)
    dataset_txt_lines.append(os.path.abspath(out_path))
    count += 1

# Write dataset.txt
with open("dataset.txt", "w") as f:
    for line in dataset_txt_lines:
        f.write(line + "\n")

print(f"\nGenerated {count} calibration samples in {OUTPUT_DIR}/")
print(f"dataset.txt written with {len(dataset_txt_lines)} entries")

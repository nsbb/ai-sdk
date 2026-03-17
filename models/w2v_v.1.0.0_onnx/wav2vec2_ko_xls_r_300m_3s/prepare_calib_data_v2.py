#!/usr/bin/env python3
"""Prepare larger calibration dataset for Korean Wav2Vec2 quantization.
Uses all available Korean + English WAV files, plus varied synthetic data.
Target: 200 samples.
"""
import numpy as np
import os
import glob
import soundfile as sf

OUTPUT_DIR = "calib_data_v2"
INPUT_LENGTH = 48000  # 3 seconds @ 16kHz
TARGET_SAMPLES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all WAV files
wav_files = []
search_dirs = [
    "/home/nsbb/travail/claude/T527/ai-sdk/models/ko_citrinet_ngc/",
    "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_base_960h_5s/data/",
    "/home/nsbb/travail/claude/T527/ai-sdk/models/deepspeech2/",
]
for d in search_dirs:
    if os.path.exists(d):
        wav_files.extend(glob.glob(os.path.join(d, "**/*.wav"), recursive=True))

print(f"Found {len(wav_files)} WAV files total")

dataset_lines = []
count = 0

for wav_path in wav_files:
    if count >= TARGET_SAMPLES:
        break
    try:
        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr != 16000:
            # Simple linear resample (approximate)
            ratio = 16000 / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)
        audio = audio.astype(np.float32)
        # Normalize (wav2vec2 feature extractor does this)
        std = np.std(audio)
        if std > 1e-7:
            audio = (audio - np.mean(audio)) / std
        else:
            continue  # Skip silent files

        # Pad or truncate
        if len(audio) < INPUT_LENGTH:
            audio = np.pad(audio, (0, INPUT_LENGTH - len(audio)))
        else:
            audio = audio[:INPUT_LENGTH]

        audio = audio.reshape(1, INPUT_LENGTH)
        out_path = os.path.join(OUTPUT_DIR, f"calib_{count:04d}.npy")
        np.save(out_path, audio)
        dataset_lines.append(os.path.abspath(out_path))
        count += 1
    except Exception as e:
        pass  # Skip problematic files

# Also use English NPY files
en_npy_dir = "/home/nsbb/travail/claude/T527/ai-sdk/models/w2v_v.1.0.0_onnx/wav2vec2_base_960h_5s/data/english_test"
if os.path.exists(en_npy_dir):
    en_npys = sorted(glob.glob(os.path.join(en_npy_dir, "*.npy")))
    for npy_path in en_npys:
        if count >= TARGET_SAMPLES:
            break
        audio = np.load(npy_path).astype(np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        if audio.shape[1] < INPUT_LENGTH:
            audio = np.pad(audio, ((0, 0), (0, INPUT_LENGTH - audio.shape[1])))
        else:
            audio = audio[:, :INPUT_LENGTH]
        out_path = os.path.join(OUTPUT_DIR, f"calib_{count:04d}.npy")
        np.save(out_path, audio)
        dataset_lines.append(os.path.abspath(out_path))
        count += 1

# Fill remaining with augmented versions (speed/volume perturbation)
existing_count = count
idx = 0
while count < TARGET_SAMPLES:
    # Load a random existing sample and augment
    src_path = dataset_lines[idx % existing_count]
    audio = np.load(src_path)

    # Random augmentation
    np.random.seed(count)
    aug_type = count % 3
    if aug_type == 0:
        # Volume change
        audio = audio * np.random.uniform(0.5, 2.0)
    elif aug_type == 1:
        # Add noise
        audio = audio + np.random.randn(*audio.shape) * 0.05
    else:
        # Slight speed change (resample)
        speed = np.random.uniform(0.9, 1.1)
        new_len = int(audio.shape[1] * speed)
        indices = np.linspace(0, audio.shape[1] - 1, new_len)
        audio_1d = np.interp(indices, np.arange(audio.shape[1]), audio[0])
        if len(audio_1d) < INPUT_LENGTH:
            audio_1d = np.pad(audio_1d, (0, INPUT_LENGTH - len(audio_1d)))
        else:
            audio_1d = audio_1d[:INPUT_LENGTH]
        audio = audio_1d.reshape(1, INPUT_LENGTH)

    # Re-normalize
    std = np.std(audio)
    if std > 1e-7:
        audio = (audio - np.mean(audio)) / std

    if audio.shape[1] != INPUT_LENGTH:
        if audio.shape[1] < INPUT_LENGTH:
            audio = np.pad(audio, ((0, 0), (0, INPUT_LENGTH - audio.shape[1])))
        else:
            audio = audio[:, :INPUT_LENGTH]

    out_path = os.path.join(OUTPUT_DIR, f"calib_{count:04d}.npy")
    np.save(out_path, audio.astype(np.float32))
    dataset_lines.append(os.path.abspath(out_path))
    count += 1
    idx += 1

# Write dataset.txt
with open("dataset_v2.txt", "w") as f:
    for line in dataset_lines:
        f.write(line + "\n")

print(f"Generated {count} calibration samples in {OUTPUT_DIR}/")
print(f"  Real audio: {existing_count}, Augmented: {count - existing_count}")
print(f"dataset_v2.txt written")

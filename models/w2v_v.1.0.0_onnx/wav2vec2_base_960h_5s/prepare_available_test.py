#!/usr/bin/env python3
"""
Prepare English test data from available WAV files.
Since LibriSpeech download is too slow, use existing audio files.
"""
import os
import struct
import shutil
import numpy as np
import soundfile as sf

OUTPUT_DIR = "data/english_test"
TARGET_SR = 16000
TARGET_LENGTH = 80000  # 5 seconds

# Available test files with ground truth (LibriSpeech speaker 1188, chapter 133604)
SAMPLES = [
    {
        "src": "data/test.wav",
        "id": "1188-133604-0000",
        "text": "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL",
    },
    {
        "src": "/home/nsbb/travail/claude/T527/ai-sdk/models/deepspeech2/data/1188-133604-0010.flac.wav",
        "id": "1188-133604-0010",
        "text": "A DARK OBJECT AGAINST A LIGHT BACKGROUND WILL APPEAR SMALLER THAN THE SAME OBJECT PLACED AGAINST A DARK BACKGROUND",
    },
    {
        "src": "/home/nsbb/travail/claude/T527/ai-sdk/models/deepspeech2/data/1188-133604-0025.flac.wav",
        "id": "1188-133604-0025",
        "text": "THIS PROCESS IS EXPERIMENTAL AND THE KEYWORDS MAY BE UPDATED AS THE LEARNING ALGORITHM IMPROVES",
    },
]

def write_wav(filepath, data, sr=16000):
    """Write float32 audio data as 16-bit PCM WAV."""
    pcm16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    num_samples = len(pcm16)
    data_size = num_samples * 2
    with open(filepath, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', sr * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm16.tobytes())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ground_truth = []

    for i, sample in enumerate(SAMPLES):
        src = sample["src"]
        if not os.path.exists(src):
            print(f"SKIP: {src} not found")
            continue

        audio, sr = sf.read(src, dtype='float32')
        duration = len(audio) / sr
        print(f"[{i}] {sample['id']}: sr={sr}, samples={len(audio)}, duration={duration:.2f}s")

        # Pad/truncate to TARGET_LENGTH
        if len(audio) >= TARGET_LENGTH:
            audio_5s = audio[:TARGET_LENGTH]
        else:
            audio_5s = np.pad(audio, (0, TARGET_LENGTH - len(audio)))

        wav_name = f"en_test_{i:04d}.wav"
        wav_path = os.path.join(OUTPUT_DIR, wav_name)
        write_wav(wav_path, audio_5s, TARGET_SR)

        npy_name = f"en_test_{i:04d}.npy"
        npy_path = os.path.join(OUTPUT_DIR, npy_name)
        np.save(npy_path, audio_5s.reshape(1, -1))

        ground_truth.append(f"{wav_name}\t{sample['text']}\t{duration:.2f}")

    # Write ground truth
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.txt")
    with open(gt_path, 'w') as f:
        f.write("# filename\tground_truth\tduration_sec\n")
        for line in ground_truth:
            f.write(line + "\n")

    print(f"\nPrepared {len(ground_truth)} samples in {OUTPUT_DIR}/")
    print(f"Ground truth: {gt_path}")

    # Also create input .dat files for vpm_run
    print("\nCreating vpm_run input .dat files...")

    input_scale = 0.002860036
    input_zp = 137

    for i in range(len(ground_truth)):
        npy_path = os.path.join(OUTPUT_DIR, f"en_test_{i:04d}.npy")
        audio = np.load(npy_path).flatten()

        # Quantize to uint8 (same as awwav2vecsdk.c)
        quantized = np.clip(np.round(audio / input_scale + input_zp), 0, 255).astype(np.uint8)

        dat_path = os.path.join(OUTPUT_DIR, f"input_{i:04d}.dat")
        quantized.tofile(dat_path)
        print(f"  {dat_path}: {len(quantized)} bytes")

    # Create vpm_run sample files
    for i in range(len(ground_truth)):
        sample_path = os.path.join(OUTPUT_DIR, f"sample_{i:04d}.txt")
        with open(sample_path, 'w') as f:
            f.write("[network]\n")
            f.write(f"/data/local/tmp/w2v_test/network_binary.nb\n")
            f.write("[input]\n")
            f.write(f"/data/local/tmp/w2v_test/input_{i:04d}.dat\n")
            f.write("[output]\n")
            f.write(f"/data/local/tmp/w2v_test/output_{i:04d}.dat\n")

    print(f"\nReady for vpm_run testing!")

if __name__ == "__main__":
    main()

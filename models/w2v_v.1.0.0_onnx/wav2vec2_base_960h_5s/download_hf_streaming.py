#!/usr/bin/env python3
"""
Download LibriSpeech test-clean samples via Hugging Face datasets streaming.
No full download needed - streams individual samples on demand.
"""
import os
import struct
import numpy as np
from datasets import load_dataset

OUTPUT_DIR = "data/english_test"
TARGET_SR = 16000
TARGET_LENGTH = 80000  # 5 seconds
NUM_SAMPLES = 50

def write_wav(filepath, data, sr=16000):
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

    print("Loading LibriSpeech test-clean via streaming...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True)

    ground_truth = []
    count = 0
    input_scale = 0.002860036
    input_zp = 137

    for sample in ds:
        if count >= NUM_SAMPLES:
            break

        audio = sample["audio"]
        sr = audio["sampling_rate"]
        data = np.array(audio["array"], dtype=np.float32)
        text = sample["text"]
        utt_id = sample.get("id", f"sample_{count}")

        duration = len(data) / sr

        # Skip very short or very long
        if duration < 1.0 or duration > 8.0:
            continue

        # Resample if needed (LibriSpeech should be 16kHz)
        if sr != TARGET_SR:
            print(f"  SKIP {utt_id}: sr={sr} (expected 16000)")
            continue

        # Pad/truncate to 5 seconds
        if len(data) >= TARGET_LENGTH:
            data_5s = data[:TARGET_LENGTH]
        else:
            data_5s = np.pad(data, (0, TARGET_LENGTH - len(data)))

        # Save WAV
        wav_name = f"en_test_{count:04d}.wav"
        wav_path = os.path.join(OUTPUT_DIR, wav_name)
        write_wav(wav_path, data_5s, TARGET_SR)

        # Save NPY
        npy_path = os.path.join(OUTPUT_DIR, f"en_test_{count:04d}.npy")
        np.save(npy_path, data_5s.reshape(1, -1))

        # Save quantized input for vpm_run
        quantized = np.clip(np.round(data_5s / input_scale + input_zp), 0, 255).astype(np.uint8)
        dat_path = os.path.join(OUTPUT_DIR, f"input_{count:04d}.dat")
        quantized.tofile(dat_path)

        # vpm_run sample file
        sample_path = os.path.join(OUTPUT_DIR, f"sample_{count:04d}.txt")
        with open(sample_path, 'w') as f:
            f.write("[network]\n/data/local/tmp/w2v_test/network_binary.nb\n")
            f.write(f"[input]\n/data/local/tmp/w2v_test/input_{count:04d}.dat\n")
            f.write(f"[output]\n/data/local/tmp/w2v_test/output_{count:04d}.dat\n")

        ground_truth.append(f"{wav_name}\t{text}\t{duration:.2f}")
        count += 1

        if count <= 5 or count % 10 == 0:
            text_short = text[:50] + "..." if len(text) > 50 else text
            print(f"  [{count:2d}/{NUM_SAMPLES}] {utt_id}: {duration:.1f}s, \"{text_short}\"")

    # Write ground truth
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.txt")
    with open(gt_path, 'w') as f:
        f.write("# filename\tground_truth\tduration_sec\n")
        for line in ground_truth:
            f.write(line + "\n")

    print(f"\nPrepared {count} samples in {OUTPUT_DIR}/")
    print(f"Ground truth: {gt_path}")

if __name__ == "__main__":
    main()

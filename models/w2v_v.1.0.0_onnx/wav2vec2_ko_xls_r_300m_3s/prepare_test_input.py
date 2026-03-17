#!/usr/bin/env python3
"""Prepare test input .dat file for Korean Wav2Vec2 vpm_run test."""
import numpy as np
import os
import glob

INPUT_LENGTH = 48000  # 3s @ 16kHz
SCALE = 0.06867171823978424
ZP = 120

def quantize_audio(audio_f32):
    """float32 → uint8 quantization."""
    quantized = np.clip(np.round(audio_f32 / SCALE) + ZP, 0, 255).astype(np.uint8)
    return quantized

# Find a Korean WAV file for testing
wav_files = []
for d in [
    "/home/nsbb/travail/claude/T527/ai-sdk/models/ko_citrinet_ngc/data",
    "/home/nsbb/travail/claude/T527/ai-sdk/models/ko_citrinet_ngc/v2_nb_fixlen/test_wavs",
]:
    if os.path.exists(d):
        wav_files.extend(sorted(glob.glob(os.path.join(d, "*.wav")))[:5])
        wav_files.extend(sorted(glob.glob(os.path.join(d, "**/*.wav"), recursive=True))[:5])

if wav_files:
    try:
        import soundfile as sf
        wav_path = wav_files[0]
        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)
        # Normalize
        audio = (audio - audio.mean()) / (np.std(audio) + 1e-7)
        print(f"Loaded: {wav_path} ({len(audio)} samples, {len(audio)/16000:.1f}s)")
    except Exception as e:
        print(f"WAV load failed: {e}, using zeros")
        audio = np.zeros(INPUT_LENGTH, dtype=np.float32)
else:
    print("No WAV files found, using zeros")
    audio = np.zeros(INPUT_LENGTH, dtype=np.float32)

# Pad/truncate to INPUT_LENGTH
if len(audio) < INPUT_LENGTH:
    audio = np.pad(audio, (0, INPUT_LENGTH - len(audio)))
else:
    audio = audio[:INPUT_LENGTH]

audio = audio.reshape(1, INPUT_LENGTH)

# Quantize
quantized = quantize_audio(audio)
print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
print(f"Quantized range: [{quantized.min()}, {quantized.max()}]")

# Save input .dat
quantized.tofile("input_0.dat")
print(f"Saved: input_0.dat ({os.path.getsize('input_0.dat')} bytes)")

# Also save float npy for ONNX comparison
np.save("test_audio.npy", audio)

# Create vpm_run sample.txt
with open("sample.txt", "w") as f:
    f.write("[network]\n")
    f.write("network_binary.nb\n")
    f.write("[input]\n")
    f.write("input_0.dat\n")
    f.write("[output]\n")
    f.write("output_0.dat\n")
print("Saved: sample.txt")

# ONNX reference decode
try:
    import onnxruntime as ort
    import json

    # Load vocab
    with open("/home/nsbb/travail/wav2vec/w2v_v.1.0.0_onnx/vocab.json") as f:
        vocab_raw = json.load(f)
    # Invert: id -> char
    vocab = {v: k for k, v in vocab_raw.items()}

    sess = ort.InferenceSession("wav2vec2_ko_3s.onnx")
    logits = sess.run(None, {"input": audio})[0]  # [1, 149, 2617]
    tokens = np.argmax(logits[0], axis=1)  # [149]

    # CTC greedy decode
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)

    pad_id = vocab_raw.get("<pad>", 2616)
    blank_ids = {0, pad_id}  # blank and pad
    text = ''.join(vocab.get(int(t), '') for t in deduped if int(t) not in blank_ids)
    # Replace | with space
    text = text.replace('|', ' ')
    print(f"\nONNX FP32 reference: '{text}'")
    print(f"Top tokens: {tokens[:20]}")
except Exception as e:
    print(f"ONNX reference failed: {e}")

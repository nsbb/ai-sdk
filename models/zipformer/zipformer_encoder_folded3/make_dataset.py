#!/usr/bin/env python3
"""
Zipformer encoder folded3 dataset maker (31 inputs, no cached_len)
- dataset0: x [1, 39, 80] mel features from real WAVs
- dataset1-30: zero-filled cache inputs (all float32)
"""
import numpy as np
import wave
import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(WORK_DIR, 'dataset')
os.makedirs(CALIB_DIR, exist_ok=True)

WAV_DIR = '/home/nsbb/travail/claude/T527/ai-sdk/models/zipformer/sherpa-onnx-streaming-zipformer-korean-2024-06-16/test_wavs'
KO_CALIB = '/home/nsbb/travail/claude/T527/ai-sdk/models/ko_citrinet_ngc/data_quickcheck/wav_calib'

CHUNK_FRAMES = 39
N_MELS = 80
CALIB_COUNT = 20

def load_wav_16k(path):
    with wave.open(path, 'rb') as wf:
        rate = wf.getframerate()
        nch = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        pcm = pcm[::nch]
    if rate != 16000:
        factor = 16000 / rate
        new_len = int(len(pcm) * factor)
        pcm = np.interp(np.linspace(0, len(pcm)-1, new_len), np.arange(len(pcm)), pcm)
    return pcm.astype(np.float32)

def extract_fbank(pcm, sr=16000, n_mels=80, n_fft=512, hop=160, chunk_frames=39):
    pcm = np.append(pcm[0], pcm[1:] - 0.97 * pcm[:-1])
    frames = []
    for i in range(0, len(pcm) - n_fft, hop):
        frame = pcm[i:i+n_fft]
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / (n_fft - 1)))
        frame = frame * window
        spec = np.abs(np.fft.rfft(frame)) ** 2
        frames.append(spec)
    if len(frames) == 0:
        return None
    stft = np.array(frames).T
    fmin, fmax = 0, sr // 2
    n_bins = n_fft // 2 + 1
    mel_min = 2595 * np.log10(1 + fmin/700)
    mel_max = 2595 * np.log10(1 + fmax/700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10**(mel_points/2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filters = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m-1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m+1]
        for k in range(f_m_minus, f_m):
            filters[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus) if f_m != f_m_minus else 0
        for k in range(f_m, f_m_plus):
            filters[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m) if f_m_plus != f_m else 0
    mel_spec = np.dot(filters, stft)
    log_mel = np.log(mel_spec + 1e-10)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True) + 1e-5
    log_mel = (log_mel - mean) / std
    chunks = []
    T = log_mel.shape[1]
    for i in range(0, T - chunk_frames + 1, chunk_frames):
        chunk = log_mel[:, i:i+chunk_frames]
        chunks.append(chunk.T.astype(np.float32))
    return chunks

wav_files = []
for d in [WAV_DIR, KO_CALIB]:
    if os.path.exists(d):
        wav_files.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.wav')])

print(f"Found {len(wav_files)} WAV files")

all_chunks = []
for wav in wav_files:
    try:
        pcm = load_wav_16k(wav)
        chunks = extract_fbank(pcm)
        if chunks:
            all_chunks.extend(chunks)
    except Exception as e:
        print(f"  Skip {wav}: {e}")

print(f"Total chunks: {len(all_chunks)}")
if len(all_chunks) > CALIB_COUNT:
    idx = np.linspace(0, len(all_chunks)-1, CALIB_COUNT, dtype=int)
    all_chunks = [all_chunks[i] for i in idx]
print(f"Using {len(all_chunks)} chunks")

# dataset0: x [1, 39, 80]
d0_dir = os.path.join(CALIB_DIR, 'dataset0')
os.makedirs(d0_dir, exist_ok=True)
with open(os.path.join(WORK_DIR, 'dataset0.txt'), 'w') as f:
    for i, chunk in enumerate(all_chunks):
        npy_path = os.path.join(d0_dir, f'calib_{i:03d}.npy')
        np.save(npy_path, chunk[np.newaxis])  # [1, 39, 80]
        f.write(npy_path + '\n')

# dataset1-30: zero-filled float32 cache inputs
# Order matches model input order (no cached_len)
CACHE_SPECS = [
    ('cached_avg_0', (2, 1, 384)),
    ('cached_avg_1', (4, 1, 384)),
    ('cached_avg_2', (3, 1, 384)),
    ('cached_avg_3', (2, 1, 384)),
    ('cached_avg_4', (4, 1, 384)),
    ('cached_key_0', (2, 64, 1, 192)),
    ('cached_key_1', (4, 32, 1, 192)),
    ('cached_key_2', (3, 16, 1, 192)),
    ('cached_key_3', (2, 8, 1, 192)),
    ('cached_key_4', (4, 32, 1, 192)),
    ('cached_val_0', (2, 64, 1, 96)),
    ('cached_val_1', (4, 32, 1, 96)),
    ('cached_val_2', (3, 16, 1, 96)),
    ('cached_val_3', (2, 8, 1, 96)),
    ('cached_val_4', (4, 32, 1, 96)),
    ('cached_val2_0', (2, 64, 1, 96)),
    ('cached_val2_1', (4, 32, 1, 96)),
    ('cached_val2_2', (3, 16, 1, 96)),
    ('cached_val2_3', (2, 8, 1, 96)),
    ('cached_val2_4', (4, 32, 1, 96)),
    ('cached_conv1_0', (2, 1, 384, 30)),
    ('cached_conv1_1', (4, 1, 384, 30)),
    ('cached_conv1_2', (3, 1, 384, 30)),
    ('cached_conv1_3', (2, 1, 384, 30)),
    ('cached_conv1_4', (4, 1, 384, 30)),
    ('cached_conv2_0', (2, 1, 384, 30)),
    ('cached_conv2_1', (4, 1, 384, 30)),
    ('cached_conv2_2', (3, 1, 384, 30)),
    ('cached_conv2_3', (2, 1, 384, 30)),
    ('cached_conv2_4', (4, 1, 384, 30)),
]

for ds_idx, (name, shape) in enumerate(CACHE_SPECS, 1):
    d_dir = os.path.join(CALIB_DIR, f'dataset{ds_idx}')
    os.makedirs(d_dir, exist_ok=True)
    with open(os.path.join(WORK_DIR, f'dataset{ds_idx}.txt'), 'w') as f:
        for i in range(len(all_chunks)):
            npy_path = os.path.join(d_dir, f'calib_{i:03d}.npy')
            np.save(npy_path, np.zeros(shape, dtype=np.float32))
            f.write(npy_path + '\n')

print(f"Created dataset0.txt - dataset30.txt ({len(all_chunks)} samples each)")
print("Done!")

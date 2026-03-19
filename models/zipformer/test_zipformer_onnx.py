#!/usr/bin/env python3
"""Test Zipformer streaming transducer with ONNX Runtime (baseline)."""

import numpy as np
import onnxruntime as ort
import time
import sys
import os

BASE = "/home/nsbb/travail/claude/T527/ai-sdk/models/zipformer"
SHERPA = os.path.join(BASE, "sherpa-onnx-streaming-zipformer-korean-2024-06-16")

ENCODER_ONNX = os.path.join(SHERPA, "encoder-epoch-99-avg-1.onnx")
DECODER_ONNX = os.path.join(SHERPA, "decoder-epoch-99-avg-1.onnx")
JOINER_ONNX = os.path.join(SHERPA, "joiner-epoch-99-avg-1.onnx")
TOKENS = os.path.join(SHERPA, "tokens.txt")
WAV_DIR = os.path.join(SHERPA, "test_wavs")

# Model config
CHUNK_FRAMES = 39      # encoder input: 39 mel frames per chunk
N_MELS = 80
SAMPLE_RATE = 16000
BLANK_ID = 0

def load_tokens(path):
    tokens = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token = parts[0]
                idx = int(parts[1])
                tokens[idx] = token
    return tokens

def compute_features(wav_path):
    """Compute log mel spectrogram features using kaldi_native_fbank."""
    import wave
    with wave.open(wav_path, 'rb') as wf:
        assert wf.getsampwidth() == 2
        assert wf.getcomptype() == 'NONE'
        sr = wf.getframerate()
        n = wf.getnframes()
        data = wf.readframes(n)

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz")

    try:
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.mel_opts.num_bins = N_MELS
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(SAMPLE_RATE, samples.tolist())
        fbank.input_finished()
        features = np.array([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    except ImportError:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=samples, sr=SAMPLE_RATE, n_fft=512, hop_length=160,
            win_length=400, n_mels=N_MELS, fmin=20, fmax=None
        )
        features = np.log(np.maximum(mel, 1e-10)).T  # [T, 80]

    return features

def init_encoder_states(encoder_sess):
    """Initialize encoder cached states (zeros) with correct dtypes."""
    feed = {}
    for inp in encoder_sess.get_inputs():
        if inp.name == "x":
            continue
        shape = inp.shape
        actual_shape = [s if isinstance(s, int) else 1 for s in shape]
        # cached_len_* are int64, everything else is float32
        if "cached_len" in inp.name:
            dtype = np.int64
        else:
            dtype = np.float32
        feed[inp.name] = np.zeros(actual_shape, dtype=dtype)
    return feed

def update_encoder_states(feed, enc_outputs, encoder_sess):
    """Map encoder outputs (new_cached_*) back to input states (cached_*)."""
    output_names = [o.name for o in encoder_sess.get_outputs()]
    for i, oname in enumerate(output_names):
        if oname == "encoder_out":
            continue
        # new_cached_len_0 -> cached_len_0, new_cached_avg_0 -> cached_avg_0, etc.
        input_name = oname.replace("new_", "")
        feed[input_name] = enc_outputs[i]

def greedy_search_onnx(wav_path, encoder_sess, decoder_sess, joiner_sess, tokens):
    """Run streaming transducer greedy search."""
    features = compute_features(wav_path)
    T = features.shape[0]

    # Pad to chunk boundary (39 frames per chunk)
    pad_len = (CHUNK_FRAMES - (T % CHUNK_FRAMES)) % CHUNK_FRAMES
    features_padded = np.pad(features, ((0, pad_len), (0, 0)))
    n_chunks = features_padded.shape[0] // CHUNK_FRAMES

    # Initialize encoder states
    feed = init_encoder_states(encoder_sess)

    t0 = time.time()

    # Process all chunks through encoder
    all_encoder_out = []
    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_FRAMES
        end = start + CHUNK_FRAMES
        feed["x"] = features_padded[start:end, :].reshape(1, CHUNK_FRAMES, N_MELS).astype(np.float32)

        enc_outputs = encoder_sess.run(None, feed)
        encoder_out = enc_outputs[0]  # [1, 8, 512]
        all_encoder_out.append(encoder_out)

        # Update cached states for next chunk
        update_encoder_states(feed, enc_outputs, encoder_sess)

    encoder_time = time.time() - t0

    if not all_encoder_out:
        return "", encoder_time, 0

    full_encoder_out = np.concatenate(all_encoder_out, axis=1)  # [1, T', 512]

    # Decoder init: context = [blank, blank]
    context = np.array([[BLANK_ID, BLANK_ID]], dtype=np.int64)
    decoder_out = decoder_sess.run(None, {"y": context})[0]  # [1, 512]

    # Greedy search through encoder output
    hyp = []
    t0_decode = time.time()

    for t in range(full_encoder_out.shape[1]):
        enc_frame = full_encoder_out[:, t:t+1, :].reshape(1, 512)  # [1, 512]

        joiner_out = joiner_sess.run(None, {
            "encoder_out": enc_frame,
            "decoder_out": decoder_out.reshape(1, 512)
        })[0]  # [1, 5000]

        y = np.argmax(joiner_out, axis=-1)[0]

        if y != BLANK_ID:
            hyp.append(y)
            context = np.array([[context[0, 1], y]], dtype=np.int64)
            decoder_out = decoder_sess.run(None, {"y": context})[0]

    decode_time = time.time() - t0_decode

    # Convert token IDs to text (BPE tokens with ▁ as word boundary)
    text = ""
    for tid in hyp:
        token = tokens.get(tid, f"<{tid}>")
        if token.startswith("\u2581"):  # ▁
            text += " " + token[1:]
        else:
            text += token
    text = text.strip()

    return text, encoder_time, decode_time

def main():
    print("Loading ONNX models...")
    encoder_sess = ort.InferenceSession(ENCODER_ONNX, providers=['CPUExecutionProvider'])
    decoder_sess = ort.InferenceSession(DECODER_ONNX, providers=['CPUExecutionProvider'])
    joiner_sess = ort.InferenceSession(JOINER_ONNX, providers=['CPUExecutionProvider'])

    tokens = load_tokens(TOKENS)
    print(f"Loaded {len(tokens)} tokens")

    # Print encoder I/O summary
    enc_inputs = encoder_sess.get_inputs()
    print(f"\nEncoder: {len(enc_inputs)} inputs, {len(encoder_sess.get_outputs())} outputs")
    print(f"  x: {enc_inputs[0].shape} ({enc_inputs[0].type})")
    print(f"  cached states: {len(enc_inputs)-1} tensors")
    print(f"  encoder_out: {encoder_sess.get_outputs()[0].shape}")

    # Load ground truth
    gt = {}
    with open(os.path.join(WAV_DIR, "trans.txt")) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                gt[parts[0]] = parts[1]

    print(f"\n{'='*80}")
    print("Zipformer Streaming Transducer ONNX Baseline")
    print(f"{'='*80}\n")

    for wav_name in sorted(gt.keys()):
        wav_path = os.path.join(WAV_DIR, wav_name)
        try:
            text, enc_time, dec_time = greedy_search_onnx(wav_path, encoder_sess, decoder_sess, joiner_sess, tokens)
            total = enc_time + dec_time
            print(f"[{wav_name}]")
            print(f"  GT:   {gt[wav_name]}")
            print(f"  PRED: {text}")
            print(f"  Time: enc={enc_time*1000:.0f}ms, dec={dec_time*1000:.0f}ms, total={total*1000:.0f}ms")
            print()
        except Exception as e:
            print(f"[{wav_name}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()

if __name__ == "__main__":
    main()

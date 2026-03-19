#!/usr/bin/env python3
"""Test Zipformer encoder NB on T527 NPU, then decode with ONNX joiner/decoder.

Strategy: Run encoder chunks on NPU (the expensive part), then use ONNX
decoder+joiner for greedy search. This tests encoder quantization quality
without the overhead of running every joiner call via adb.
"""

import numpy as np
import subprocess
import os
import sys
import time
import json
import tempfile
import shutil

BASE = "/home/nsbb/travail/claude/T527/ai-sdk/models/zipformer"
BUNDLE = os.path.join(BASE, "bundle_uint8")
SHERPA = os.path.join(BASE, "sherpa-onnx-streaming-zipformer-korean-2024-06-16")
WAV_DIR = os.path.join(SHERPA, "test_wavs")
TOKENS_PATH = os.path.join(SHERPA, "tokens.txt")

WIN_ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEVICE_DIR = "/data/local/tmp/zipformer_test"
VPM_RUN = "/data/local/tmp/vpm_run_aarch64"

CHUNK_FRAMES = 39
N_MELS = 80
SAMPLE_RATE = 16000
BLANK_ID = 0

# Encoder quant params
ENC_INPUT_SCALE = 0.021361934021115303
ENC_INPUT_ZP = 125
ENC_OUTPUT_SCALE = 0.01050473377108574
ENC_OUTPUT_ZP = 116

def load_tokens(path):
    tokens = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tokens[int(parts[1])] = parts[0]
    return tokens

def compute_features(wav_path):
    import wave
    with wave.open(wav_path, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        data = wf.readframes(n)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    assert sr == SAMPLE_RATE

    import kaldi_native_fbank as knf
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.mel_opts.num_bins = N_MELS
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, samples.tolist())
    fbank.input_finished()
    return np.array([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])

def adb(*args):
    cmd = [WIN_ADB] + list(args)
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return r.stdout.decode('utf-8', errors='replace'), r.returncode

def quantize_u8(data, scale, zp):
    return np.clip(np.round(data / scale + zp), 0, 255).astype(np.uint8)

def dequantize_u8(data, scale, zp):
    return (data.astype(np.float32) - zp) * scale

def run_encoder_chunks_npu(features_padded, n_chunks, enc_meta, work_dir):
    """Run all encoder chunks on T527 NPU.

    For each chunk: push inputs → vpm_run → pull output.
    No state passing (each chunk independent).

    Returns: list of dequantized encoder outputs [1, 8, 512] each
    """
    device_dir = f"{DEVICE_DIR}/encoder"
    adb("shell", f"rm -rf {DEVICE_DIR} && mkdir -p {device_dir}")

    # Push encoder NB once
    nb_local = os.path.join(BUNDLE, "encoder", "network_binary.nb")
    nb_device = f"{device_dir}/network_binary.nb"
    adb("push", nb_local, nb_device)

    # Prepare zero cached state files (same for all chunks)
    input_order = list(enc_meta["Inputs"].keys())
    state_files_device = []

    for key in input_order:
        info = enc_meta["Inputs"][key]
        name = info["name"]
        if name == "x":
            continue
        shape = info["shape"]
        scale = info["quantize"]["scale"]
        zp = info["quantize"]["zero_point"]
        total_size = 1
        for s in shape:
            total_size *= s
        # Zero float = zp in uint8
        local_path = os.path.join(work_dir, f"{name}.dat")
        np.full(total_size, zp, dtype=np.uint8).tofile(local_path)
        device_path = f"{device_dir}/{name}.dat"
        adb("push", local_path, device_path)
        state_files_device.append(device_path)

    all_encoder_out = []
    total_npu_ms = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_FRAMES
        end = start + CHUNK_FRAMES
        chunk = features_padded[start:end, :]

        # Quantize and push mel features
        x_q = quantize_u8(chunk.reshape(1, CHUNK_FRAMES, N_MELS), ENC_INPUT_SCALE, ENC_INPUT_ZP)
        x_local = os.path.join(work_dir, "x.dat")
        x_q.tofile(x_local)
        x_device = f"{device_dir}/x.dat"
        adb("push", x_local, x_device)

        # Build sample.txt
        input_list = [x_device] + state_files_device
        output_device = f"{device_dir}/output_0.dat"

        sample_content = f"[network]\n{nb_device}\n[input]\n"
        for inp in input_list:
            sample_content += f"{inp}\n"
        sample_content += f"[output]\n{output_device}\n"

        sample_local = os.path.join(work_dir, "sample.txt")
        with open(sample_local, 'w') as f:
            f.write(sample_content)
        adb("push", sample_local, f"{device_dir}/sample.txt")

        # Run vpm_run
        out, rc = adb("shell",
            f"cd {device_dir} && LD_LIBRARY_PATH=/vendor/lib64 {VPM_RUN} -s sample.txt -b 0")

        # Parse timing from vpm_run output
        for line in out.split('\n'):
            if 'time(ms)' in line.lower():
                try:
                    ms = float(line.split()[-1])
                    total_npu_ms += ms
                except:
                    pass

        # Pull and dequantize output
        out_local = os.path.join(work_dir, f"enc_out_{chunk_idx}.dat")
        adb("pull", output_device, out_local)

        raw = np.fromfile(out_local, dtype=np.uint8)
        expected = 1 * 8 * 512
        if raw.size >= expected:
            enc_out = dequantize_u8(raw[:expected].reshape(1, 8, 512), ENC_OUTPUT_SCALE, ENC_OUTPUT_ZP)
            all_encoder_out.append(enc_out)
            nz = np.count_nonzero(enc_out)
            print(f"    chunk {chunk_idx}: range=[{enc_out.min():.3f},{enc_out.max():.3f}] "
                  f"nonzero={nz}/{enc_out.size}")
        else:
            print(f"    chunk {chunk_idx}: BAD output size {raw.size} (expected {expected})")

    return all_encoder_out, total_npu_ms

def greedy_search_with_onnx_decoder(full_encoder_out, decoder_sess, joiner_sess, tokens):
    """Greedy search using ONNX decoder+joiner on dequantized NPU encoder output."""
    context = np.array([[BLANK_ID, BLANK_ID]], dtype=np.int64)
    decoder_out = decoder_sess.run(None, {"y": context})[0]

    hyp = []
    for t in range(full_encoder_out.shape[1]):
        enc_frame = full_encoder_out[:, t, :].reshape(1, 512)
        joiner_out = joiner_sess.run(None, {
            "encoder_out": enc_frame,
            "decoder_out": decoder_out.reshape(1, 512)
        })[0]

        y = np.argmax(joiner_out, axis=-1)[0]
        if y != BLANK_ID:
            hyp.append(y)
            context = np.array([[context[0, 1], y]], dtype=np.int64)
            decoder_out = decoder_sess.run(None, {"y": context})[0]

    text = ""
    for tid in hyp:
        token = tokens.get(tid, f"<{tid}>")
        if token.startswith("\u2581"):
            text += " " + token[1:]
        else:
            text += token
    return text.strip()

def main():
    import onnxruntime as ort

    print("Zipformer NPU Encoder Test")
    print("=" * 60)
    print("Strategy: Encoder on T527 NPU, Decoder+Joiner on ONNX (server)")
    print()

    tokens = load_tokens(TOKENS_PATH)
    with open(os.path.join(BUNDLE, "encoder", "nbg_meta.json")) as f:
        enc_meta = json.load(f)

    # Load ONNX decoder + joiner
    decoder_sess = ort.InferenceSession(
        os.path.join(SHERPA, "decoder-epoch-99-avg-1.onnx"),
        providers=['CPUExecutionProvider'])
    joiner_sess = ort.InferenceSession(
        os.path.join(SHERPA, "joiner-epoch-99-avg-1.onnx"),
        providers=['CPUExecutionProvider'])

    # Load ground truth
    gt = {}
    with open(os.path.join(WAV_DIR, "trans.txt")) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                gt[parts[0]] = parts[1]

    work_dir = tempfile.mkdtemp(prefix="zipformer_npu_")

    for wav_name in sorted(gt.keys()):
        wav_path = os.path.join(WAV_DIR, wav_name)
        print(f"\n[{wav_name}]")

        features = compute_features(wav_path)
        T = features.shape[0]
        pad_len = (CHUNK_FRAMES - (T % CHUNK_FRAMES)) % CHUNK_FRAMES
        features_padded = np.pad(features, ((0, pad_len), (0, 0)))
        n_chunks = features_padded.shape[0] // CHUNK_FRAMES
        print(f"  {T} frames → {n_chunks} chunks")

        try:
            # Run encoder on NPU
            enc_outputs, npu_ms = run_encoder_chunks_npu(
                features_padded, n_chunks, enc_meta, work_dir)

            if not enc_outputs:
                print("  No encoder output!")
                continue

            full_enc = np.concatenate(enc_outputs, axis=1)
            print(f"  Encoder NPU: {full_enc.shape}, {npu_ms:.0f}ms total")

            # Greedy search with ONNX decoder+joiner
            text = greedy_search_with_onnx_decoder(
                full_enc, decoder_sess, joiner_sess, tokens)

            print(f"  GT:   {gt[wav_name]}")
            print(f"  PRED: {text}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    main()

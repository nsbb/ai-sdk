#!/usr/bin/env python3
"""Compare int16 NPU encoder output with ONNX float baseline for one chunk."""

import numpy as np
import onnxruntime as ort
import json
import os
import subprocess
import tempfile

BASE = "/home/nsbb/travail/claude/T527/ai-sdk/models/zipformer"
SHERPA = os.path.join(BASE, "sherpa-onnx-streaming-zipformer-korean-2024-06-16")
NB_DIR = os.path.join(BASE, "zipformer_encoder_folded4/wksp/encoder_with_states_v6_int16_fixed_nbg_unify_nbg_unify")

ENCODER_ONNX = os.path.join(SHERPA, "encoder-epoch-99-avg-1.onnx")
WAV_DIR = os.path.join(SHERPA, "test_wavs")

WIN_ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEVICE_DIR = "/data/local/tmp/zipformer_npu_int16"
VPM_RUN = "/data/local/tmp/vpm_run_aarch64"

CHUNK_FRAMES = 39
N_MELS = 80
SAMPLE_RATE = 16000


def compute_features(wav_path):
    import wave
    with wave.open(wav_path, 'rb') as wf:
        data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

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


def quantize_i16(data, fl):
    return np.clip(np.round(data * (2 ** fl)), -32768, 32767).astype(np.int16)


def dequantize_i16(data, fl):
    return data.astype(np.float32) / (2 ** fl)


def main():
    with open(os.path.join(NB_DIR, "nbg_meta.json")) as f:
        enc_meta = json.load(f)

    # Load ONNX encoder
    encoder_sess = ort.InferenceSession(ENCODER_ONNX, providers=['CPUExecutionProvider'])

    # Get first test wav
    wav_path = os.path.join(WAV_DIR, "0.wav")
    features = compute_features(wav_path)
    T = features.shape[0]
    pad_len = (CHUNK_FRAMES - (T % CHUNK_FRAMES)) % CHUNK_FRAMES
    features_padded = np.pad(features, ((0, pad_len), (0, 0)))

    # Run ONNX encoder for first chunk with zero states
    feed = {}
    for inp in encoder_sess.get_inputs():
        if inp.name == "x":
            feed["x"] = features_padded[:CHUNK_FRAMES].reshape(1, CHUNK_FRAMES, N_MELS).astype(np.float32)
        else:
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            dtype = np.int64 if "cached_len" in inp.name else np.float32
            feed[inp.name] = np.zeros(shape, dtype=dtype)

    onnx_outputs = encoder_sess.run(None, feed)
    onnx_enc_out = onnx_outputs[0]  # [1, 8, 512]

    print(f"ONNX encoder out: shape={onnx_enc_out.shape}")
    print(f"  range=[{onnx_enc_out.min():.4f}, {onnx_enc_out.max():.4f}]")
    print(f"  nonzero={np.count_nonzero(onnx_enc_out)}/{onnx_enc_out.size}")

    # Run NPU encoder for first chunk with zero states
    work_dir = tempfile.mkdtemp(prefix="zipformer_cmp_")
    device_enc = f"{DEVICE_DIR}/encoder"
    adb("shell", f"rm -rf {DEVICE_DIR} && mkdir -p {device_enc}")

    nb_path = os.path.join(NB_DIR, "network_binary.nb")
    adb("push", nb_path, f"{device_enc}/network_binary.nb")

    input_order = list(enc_meta["Inputs"].keys())
    output_order = list(enc_meta["Outputs"].keys())

    # Quantize and push mel features
    x_fl = enc_meta["Inputs"][input_order[0]]["quantize"]["fl"]
    x_data = quantize_i16(features_padded[:CHUNK_FRAMES].reshape(1, CHUNK_FRAMES, N_MELS), x_fl)
    x_local = os.path.join(work_dir, "x.dat")
    x_data.tofile(x_local)
    adb("push", x_local, f"{device_enc}/x.dat")

    # Push zero state files
    input_files_device = [f"{device_enc}/x.dat"]
    for ikey in input_order:
        iinfo = enc_meta["Inputs"][ikey]
        if iinfo["name"] == "x":
            continue
        shape = iinfo["shape"]
        total = 1
        for s in shape:
            total *= s
        local_path = os.path.join(work_dir, f"{iinfo['name']}.dat")
        np.zeros(total, dtype=np.int16).tofile(local_path)
        device_path = f"{device_enc}/{iinfo['name']}.dat"
        adb("push", local_path, device_path)
        input_files_device.append(device_path)

    # Build sample.txt
    output_files_device = [f"{device_enc}/output_{oi}.dat" for oi in range(len(output_order))]

    sample = f"[network]\n{device_enc}/network_binary.nb\n[input]\n"
    for f in input_files_device:
        sample += f + "\n"
    sample += "[output]\n"
    for f in output_files_device:
        sample += f + "\n"

    sample_local = os.path.join(work_dir, "sample.txt")
    with open(sample_local, 'w') as f:
        f.write(sample)
    adb("push", sample_local, f"{device_enc}/sample.txt")

    # Run NPU
    out, rc = adb("shell",
        f"cd {device_enc} && LD_LIBRARY_PATH=/vendor/lib64 {VPM_RUN} -s sample.txt -b 0")
    print(f"\nvpm_run output: {out.strip()[:200]}")

    # Pull encoder_out
    out0_local = os.path.join(work_dir, "enc_out_0.dat")
    adb("pull", f"{device_enc}/output_0.dat", out0_local)

    raw = np.fromfile(out0_local, dtype=np.int16)
    oinfo = enc_meta["Outputs"][output_order[0]]
    ofl = oinfo["quantize"]["fl"]
    expected = 1
    for s in oinfo["shape"]:
        expected *= s

    npu_enc_out = dequantize_i16(raw[:expected].reshape(oinfo["shape"]), ofl)

    print(f"\nNPU encoder out: shape={npu_enc_out.shape}")
    print(f"  range=[{npu_enc_out.min():.4f}, {npu_enc_out.max():.4f}]")
    print(f"  nonzero={np.count_nonzero(npu_enc_out)}/{npu_enc_out.size}")

    # Compare
    onnx_flat = onnx_enc_out.flatten()
    npu_flat = npu_enc_out.flatten()

    correlation = np.corrcoef(onnx_flat, npu_flat)[0, 1]
    cosine = np.dot(onnx_flat, npu_flat) / (np.linalg.norm(onnx_flat) * np.linalg.norm(npu_flat) + 1e-10)
    rmse = np.sqrt(np.mean((onnx_flat - npu_flat) ** 2))
    max_err = np.max(np.abs(onnx_flat - npu_flat))

    print(f"\nComparison (chunk 0):")
    print(f"  Pearson correlation: {correlation:.4f}")
    print(f"  Cosine similarity:   {cosine:.4f}")
    print(f"  RMSE:                {rmse:.4f}")
    print(f"  Max error:           {max_err:.4f}")

    # Per-frame comparison
    print(f"\nPer-frame correlation (8 frames):")
    for t in range(min(8, onnx_enc_out.shape[1])):
        onnx_frame = onnx_enc_out[0, t, :]
        npu_frame = npu_enc_out[0, t, :]
        corr = np.corrcoef(onnx_frame, npu_frame)[0, 1]
        cos = np.dot(onnx_frame, npu_frame) / (np.linalg.norm(onnx_frame) * np.linalg.norm(npu_frame) + 1e-10)
        print(f"  frame {t}: corr={corr:.4f} cos={cos:.4f}")

    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare PCQ int8 NPU encoder output with ONNX float baseline."""

import numpy as np
import onnxruntime as ort
import json
import os
import subprocess
import tempfile
import shutil

BASE = "/home/nsbb/travail/claude/T527/ai-sdk/models/zipformer"
SHERPA = os.path.join(BASE, "sherpa-onnx-streaming-zipformer-korean-2024-06-16")
NB_DIR = os.path.join(BASE, "zipformer_encoder_folded4/wksp/encoder_with_states_v6_pcq_fixed_nbg_unify_nbg_unify")
ENCODER_ONNX = os.path.join(SHERPA, "encoder-epoch-99-avg-1.onnx")
WAV_DIR = os.path.join(SHERPA, "test_wavs")

WIN_ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEVICE_DIR = "/data/local/tmp/zipformer_npu_pcq"
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


def quantize_i8(data, scale, zp):
    return np.clip(np.round(data / scale + zp), -128, 127).astype(np.int8)


def dequantize_i8(data, scale, zp):
    return (data.astype(np.float32) - zp) * scale


def main():
    with open(os.path.join(NB_DIR, "nbg_meta.json")) as f:
        enc_meta = json.load(f)

    encoder_sess = ort.InferenceSession(ENCODER_ONNX, providers=['CPUExecutionProvider'])

    wav_path = os.path.join(WAV_DIR, "0.wav")
    features = compute_features(wav_path)
    pad_len = (CHUNK_FRAMES - (features.shape[0] % CHUNK_FRAMES)) % CHUNK_FRAMES
    features_padded = np.pad(features, ((0, pad_len), (0, 0)))

    # ONNX encoder chunk 0
    feed = {}
    for inp in encoder_sess.get_inputs():
        if inp.name == "x":
            feed["x"] = features_padded[:CHUNK_FRAMES].reshape(1, CHUNK_FRAMES, N_MELS).astype(np.float32)
        else:
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            dtype = np.int64 if "cached_len" in inp.name else np.float32
            feed[inp.name] = np.zeros(shape, dtype=dtype)

    onnx_enc_out = encoder_sess.run(None, feed)[0]
    print(f"ONNX: shape={onnx_enc_out.shape} range=[{onnx_enc_out.min():.4f}, {onnx_enc_out.max():.4f}]")

    # NPU encoder chunk 0
    work_dir = tempfile.mkdtemp(prefix="zipformer_pcq_")
    device_enc = f"{DEVICE_DIR}/encoder"
    adb("shell", f"rm -rf {DEVICE_DIR} && mkdir -p {device_enc}")
    adb("push", os.path.join(NB_DIR, "network_binary.nb"), f"{device_enc}/network_binary.nb")

    input_order = list(enc_meta["Inputs"].keys())
    output_order = list(enc_meta["Outputs"].keys())

    # Quantize mel input
    x_q = enc_meta["Inputs"][input_order[0]]["quantize"]
    x_data = quantize_i8(features_padded[:CHUNK_FRAMES].reshape(1, CHUNK_FRAMES, N_MELS),
                         x_q["scale"], x_q["zero_point"])
    x_local = os.path.join(work_dir, "x.dat")
    x_data.tofile(x_local)
    adb("push", x_local, f"{device_enc}/x.dat")

    # Push zero state files
    input_files = [f"{device_enc}/x.dat"]
    for ikey in input_order:
        iinfo = enc_meta["Inputs"][ikey]
        if iinfo["name"] == "x":
            continue
        total = 1
        for s in iinfo["shape"]:
            total *= s
        zp = iinfo["quantize"]["zero_point"]
        local_path = os.path.join(work_dir, f"{iinfo['name']}.dat")
        np.full(total, zp, dtype=np.int8).tofile(local_path)
        adb("push", local_path, f"{device_enc}/{iinfo['name']}.dat")
        input_files.append(f"{device_enc}/{iinfo['name']}.dat")

    # Build sample.txt
    output_files = [f"{device_enc}/output_{oi}.dat" for oi in range(len(output_order))]
    sample = f"[network]\n{device_enc}/network_binary.nb\n[input]\n"
    for f in input_files:
        sample += f + "\n"
    sample += "[output]\n"
    for f in output_files:
        sample += f + "\n"
    sample_local = os.path.join(work_dir, "sample.txt")
    with open(sample_local, 'w') as f:
        f.write(sample)
    adb("push", sample_local, f"{device_enc}/sample.txt")

    out, rc = adb("shell",
        f"cd {device_enc} && LD_LIBRARY_PATH=/vendor/lib64 {VPM_RUN} -s sample.txt -b 0")
    print(f"vpm_run: {out.strip()[:150]}")

    # Pull and dequantize encoder_out
    out0_local = os.path.join(work_dir, "enc_out.dat")
    adb("pull", f"{device_enc}/output_0.dat", out0_local)

    raw = np.fromfile(out0_local, dtype=np.int8)
    oinfo = enc_meta["Outputs"][output_order[0]]
    oq = oinfo["quantize"]
    expected = 1
    for s in oinfo["shape"]:
        expected *= s

    npu_enc_out = dequantize_i8(raw[:expected].reshape(oinfo["shape"]), oq["scale"], oq["zero_point"])
    print(f"NPU:  shape={npu_enc_out.shape} range=[{npu_enc_out.min():.4f}, {npu_enc_out.max():.4f}]")

    # Compare
    onnx_flat = onnx_enc_out.flatten()
    npu_flat = npu_enc_out.flatten()
    correlation = np.corrcoef(onnx_flat, npu_flat)[0, 1]
    cosine = np.dot(onnx_flat, npu_flat) / (np.linalg.norm(onnx_flat) * np.linalg.norm(npu_flat) + 1e-10)
    rmse = np.sqrt(np.mean((onnx_flat - npu_flat) ** 2))

    print(f"\nPCQ int8 vs ONNX float (chunk 0):")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Cosine sim:  {cosine:.4f}")
    print(f"  RMSE:        {rmse:.4f}")

    print(f"\nPer-frame:")
    for t in range(min(8, onnx_enc_out.shape[1])):
        corr = np.corrcoef(onnx_enc_out[0, t, :], npu_enc_out[0, t, :])[0, 1]
        print(f"  frame {t}: corr={corr:.4f}")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

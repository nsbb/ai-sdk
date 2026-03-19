#!/usr/bin/env python3
"""Test Zipformer full NPU pipeline with int16 encoder and state passing.

Encoder (int16, 31 outputs) on T527 NPU + ONNX decoder/joiner for greedy search.
State outputs from each encoder chunk are passed back as inputs to the next chunk.
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
SHERPA = os.path.join(BASE, "sherpa-onnx-streaming-zipformer-korean-2024-06-16")
NB_DIR = os.path.join(BASE, "zipformer_encoder_folded4/wksp/encoder_with_states_v6_int16_fixed_nbg_unify_nbg_unify")
WAV_DIR = os.path.join(SHERPA, "test_wavs")
TOKENS_PATH = os.path.join(SHERPA, "tokens.txt")

WIN_ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEVICE_DIR = "/data/local/tmp/zipformer_npu_int16"
VPM_RUN = "/data/local/tmp/vpm_run_aarch64"

CHUNK_FRAMES = 39
N_MELS = 80
SAMPLE_RATE = 16000
BLANK_ID = 0


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
    """Quantize float to int16: i16_val = round(float_val * 2^fl)"""
    return np.clip(np.round(data * (2 ** fl)), -32768, 32767).astype(np.int16)


def dequantize_i16(data, fl):
    """Dequantize int16 to float: float_val = i16_val / 2^fl"""
    return data.astype(np.float32) / (2 ** fl)


def build_state_mapping(enc_meta):
    """Build mapping from output index to input key for state passing.

    NB outputs (after encoder_out) follow per-stack order:
      avg_N, key_N, val_N, val2_N, conv1_N, conv2_N  for N=0,1,2,3,4
    NB inputs follow per-type order:
      x, avg_0..4, key_0..4, val_0..4, val2_0..4, conv1_0..4, conv2_0..4
    """
    input_order = list(enc_meta["Inputs"].keys())
    input_by_name = {enc_meta["Inputs"][k]["name"]: k for k in input_order}

    state_types = ["cached_avg", "cached_key", "cached_val",
                   "cached_val2", "cached_conv1", "cached_conv2"]
    n_stacks = 5

    output_to_input = {}  # output_idx -> input_key
    oi = 1  # skip output[0] = encoder_out
    for stack in range(n_stacks):
        for stype in state_types:
            input_name = f"{stype}_{stack}"
            if input_name in input_by_name:
                output_to_input[oi] = input_by_name[input_name]
            oi += 1

    return output_to_input


def run_encoder_with_states(features_padded, n_chunks, enc_meta, work_dir):
    """Run encoder chunks on NPU with state passing between chunks (int16)."""
    device_enc = f"{DEVICE_DIR}/encoder"
    adb("shell", f"rm -rf {DEVICE_DIR} && mkdir -p {device_enc}")

    # Push encoder NB
    nb_path = os.path.join(NB_DIR, "network_binary.nb")
    adb("push", nb_path, f"{device_enc}/network_binary.nb")

    input_order = list(enc_meta["Inputs"].keys())
    output_order = list(enc_meta["Outputs"].keys())

    # Build state mapping
    output_to_input = build_state_mapping(enc_meta)
    print(f"  State mapping: {len(output_to_input)} outputs -> inputs")

    # Initialize state buffers (int16, zero states)
    state_buffers = {}
    for ikey in input_order:
        iinfo = enc_meta["Inputs"][ikey]
        if iinfo["name"] == "x":
            continue
        shape = iinfo["shape"]
        total = 1
        for s in shape:
            total *= s
        # For int16 dynamic_fixed_point, zero float = zero int16
        state_buffers[ikey] = np.zeros(total, dtype=np.int16)

    all_encoder_out = []
    total_npu_ms = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_FRAMES
        chunk = features_padded[start:start+CHUNK_FRAMES, :]

        # Quantize mel input to int16
        x_fl = enc_meta["Inputs"][input_order[0]]["quantize"]["fl"]
        x_data = quantize_i16(chunk.reshape(1, CHUNK_FRAMES, N_MELS), x_fl)
        x_local = os.path.join(work_dir, "x.dat")
        x_data.tofile(x_local)
        adb("push", x_local, f"{device_enc}/x.dat")

        # Push state files
        input_files_device = [f"{device_enc}/x.dat"]
        for ikey in input_order:
            iinfo = enc_meta["Inputs"][ikey]
            if iinfo["name"] == "x":
                continue
            local_path = os.path.join(work_dir, f"{iinfo['name']}.dat")
            state_buffers[ikey].tofile(local_path)
            device_path = f"{device_enc}/{iinfo['name']}.dat"
            adb("push", local_path, device_path)
            input_files_device.append(device_path)

        # Build sample.txt with all outputs
        output_files_device = []
        for oi in range(len(output_order)):
            output_files_device.append(f"{device_enc}/output_{oi}.dat")

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

        # Parse timing
        for line in out.split('\n'):
            if 'inference time' in line.lower():
                try:
                    ms = float(line.split('=')[1].split('us')[0]) / 1000
                    total_npu_ms += ms
                except:
                    pass

        # Pull encoder_out (output 0) — int16
        out0_local = os.path.join(work_dir, f"enc_out_{chunk_idx}.dat")
        adb("pull", f"{device_enc}/output_0.dat", out0_local)

        raw = np.fromfile(out0_local, dtype=np.int16)
        oinfo = enc_meta["Outputs"][output_order[0]]
        expected = 1
        for s in oinfo["shape"]:
            expected *= s

        if raw.size >= expected:
            ofl = oinfo["quantize"]["fl"]
            enc_out = dequantize_i16(raw[:expected].reshape(oinfo["shape"]), ofl)
            all_encoder_out.append(enc_out)
        else:
            print(f"    chunk {chunk_idx}: BAD output size {raw.size}")
            continue

        # Pull state outputs and update state buffers for next chunk
        for oi, ikey in output_to_input.items():
            out_local = os.path.join(work_dir, f"state_out_{oi}.dat")
            adb("pull", f"{device_enc}/output_{oi}.dat", out_local)

            oinfo_state = enc_meta["Outputs"][output_order[oi]]
            ofl = oinfo_state["quantize"]["fl"]

            # Read int16 output
            raw_state = np.fromfile(out_local, dtype=np.int16)

            # Dequantize output with output fl, then requantize for input fl
            state_float = dequantize_i16(raw_state, ofl)

            iinfo = enc_meta["Inputs"][ikey]
            ifl = iinfo["quantize"]["fl"]

            state_buffers[ikey] = quantize_i16(state_float, ifl)

        nz = np.count_nonzero(enc_out)
        print(f"    chunk {chunk_idx}: range=[{enc_out.min():.3f},{enc_out.max():.3f}] "
              f"nonzero={nz}/{enc_out.size}")

    return all_encoder_out, total_npu_ms


def greedy_search(full_encoder_out, decoder_sess, joiner_sess, tokens):
    """Greedy search using ONNX decoder+joiner."""
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

    print("Zipformer NPU INT16 Pipeline Test (with state passing)")
    print("=" * 60)

    tokens = load_tokens(TOKENS_PATH)

    with open(os.path.join(NB_DIR, "nbg_meta.json")) as f:
        enc_meta = json.load(f)

    print(f"Encoder NB: {len(enc_meta['Inputs'])} inputs, {len(enc_meta['Outputs'])} outputs")
    print(f"NB: int16 dynamic_fixed_point (118MB)")

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

    work_dir = tempfile.mkdtemp(prefix="zipformer_npu_int16_")

    total_cer_chars = 0
    total_gt_chars = 0

    for wav_name in sorted(gt.keys()):
        wav_path = os.path.join(WAV_DIR, wav_name)
        print(f"\n[{wav_name}]")

        features = compute_features(wav_path)
        T = features.shape[0]
        pad_len = (CHUNK_FRAMES - (T % CHUNK_FRAMES)) % CHUNK_FRAMES
        features_padded = np.pad(features, ((0, pad_len), (0, 0)))
        n_chunks = features_padded.shape[0] // CHUNK_FRAMES
        print(f"  {T} frames -> {n_chunks} chunks")

        try:
            enc_outputs, npu_ms = run_encoder_with_states(
                features_padded, n_chunks, enc_meta, work_dir)

            if not enc_outputs:
                print("  No encoder output!")
                continue

            full_enc = np.concatenate(enc_outputs, axis=1)
            print(f"  Encoder: {full_enc.shape}, NPU time={npu_ms:.0f}ms")

            text = greedy_search(full_enc, decoder_sess, joiner_sess, tokens)

            print(f"  GT:   {gt[wav_name]}")
            print(f"  PRED: {text}")

            # Simple CER
            gt_chars = list(gt[wav_name].replace(" ", ""))
            pred_chars = list(text.replace(" ", ""))
            n, m = len(gt_chars), len(pred_chars)
            dp = [[0]*(m+1) for _ in range(n+1)]
            for i in range(n+1):
                dp[i][0] = i
            for j in range(m+1):
                dp[0][j] = j
            for i in range(1, n+1):
                for j in range(1, m+1):
                    if gt_chars[i-1] == pred_chars[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            cer = dp[n][m] / max(n, 1) * 100
            print(f"  CER: {cer:.1f}%")
            total_cer_chars += dp[n][m]
            total_gt_chars += n

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if total_gt_chars > 0:
        overall_cer = total_cer_chars / total_gt_chars * 100
        print(f"\n{'='*60}")
        print(f"Overall CER: {overall_cer:.2f}%")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

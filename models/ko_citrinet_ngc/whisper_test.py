import os
import re
import json
import time
import glob
import shutil
import subprocess
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import soundfile as sf
import torch
import onnxruntime as ort
import nemo.collections.asr as nemo_asr
from pathlib import Path

try:
    import nlptutti as metrics
except ImportError:
    metrics = None

MODEL_NEMO = "/nas02/geonhui83/stt/citrinet_korean/Citrinet-1024-gamma-0.25_spe-2048_ko-KR_Riva-ASR-SET-1.0.nemo"
MODEL_ONNX = "/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/work/citrinet_v2_fixlen.onnx"
MODEL_JSON = "/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/work/citrinet_v2_fixlen.json"
MODEL_DATA = "/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/work/citrinet_v2_fixlen.data"
MODEL_QUANT = "/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/work/citrinet_v2_fixlen_int8.quantize"
INPUTMETA_TEMPLATE = "/nas02/geonhui83/stt/new_citirinet/ko_citrinet_ngc/v2_nb_fixlen/work/citrinet_v2_fixlen_inputmeta.yml"
PEGASUS_BIN = "/nas02/geonhui83/T527_toolkit/acuity-toolkit-binary-6.12.0/bin/pegasus"
NB_WORKDIR = "/tmp/citrinet_nb_infer"
INFER_BACKEND = os.environ.get("INFER_BACKEND", "onnx").strip().lower()

TIME_FRAMES = 300
TARGET_SR = 16000
NUM_CLASSES = 2049
NUM_FRAMES = 38

asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path=MODEL_NEMO,
    map_location="cpu",
)
asr_model.eval()
preprocessor = asr_model.preprocessor
tokenizer = asr_model.tokenizer
blank_id = asr_model.decoder.num_classes_with_blank - 1
onnx_sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])


def linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    ratio = float(dst_sr) / float(src_sr)
    out_len = max(1, int(round(len(audio) * ratio)))
    x0 = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x1 = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    return np.interp(x1, x0, audio).astype(np.float32)


def ctc_decode_ids(argmax_ids: np.ndarray, blank: int):
    out = []
    prev = None
    for x in argmax_ids.tolist():
        xi = int(x)
        if xi != blank and xi != prev:
            out.append(xi)
        prev = xi
    return out


def wav_to_model_input(wav_path):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = linear_resample(audio.astype(np.float32), sr, TARGET_SR)

    with torch.no_grad():
        sig = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        ln = torch.tensor([sig.shape[1]], dtype=torch.long)
        feat, _ = preprocessor(input_signal=sig, length=ln)  # [1,80,T]

        t = feat.shape[2]
        if t > TIME_FRAMES:
            feat = feat[:, :, :TIME_FRAMES]
        elif t < TIME_FRAMES:
            feat = torch.nn.functional.pad(feat, (0, TIME_FRAMES - t))

        x = feat.unsqueeze(2).cpu().numpy().astype(np.float32)  # [1,80,1,300]
    return x


def decode_logits_to_text(y):
    argmax_ids = y[0, :, 0, :].T.argmax(axis=1).astype(np.int64)
    token_ids = ctc_decode_ids(argmax_ids, blank=blank_id)
    return tokenizer.ids_to_text(token_ids)


def nb_inference_from_input(x):
    work_dir = Path(NB_WORKDIR) / f"run_{os.getpid()}_{int(time.time() * 1000)}"
    infer_dir = work_dir / "infer_out"
    try:
        infer_dir.mkdir(parents=True, exist_ok=True)

        npy_path = work_dir / "input.npy"
        np.save(npy_path, x)

        one_list = work_dir / "one_list.txt"
        one_list.write_text(str(npy_path) + "\n", encoding="utf-8")

        inputmeta_path = work_dir / "one_inputmeta.yml"
        lines = Path(INPUTMETA_TEMPLATE).read_text(encoding="utf-8").splitlines()
        patched = []
        for line in lines:
            s = line.strip()
            indent = line[: len(line) - len(line.lstrip())]
            if s.startswith("path: "):
                patched.append(f"{indent}path: {one_list}")
            elif s.startswith("- path: "):
                patched.append(f"{indent}- path: {one_list}")
            else:
                patched.append(line)
        inputmeta_path.write_text("\n".join(patched) + "\n", encoding="utf-8")

        cmd = [
            PEGASUS_BIN,
            "inference",
            "--model",
            MODEL_JSON,
            "--model-data",
            MODEL_DATA,
            "--model-quantize",
            MODEL_QUANT,
            "--batch-size",
            "1",
            "--iterations",
            "1",
            "--device",
            "CPU",
            "--with-input-meta",
            str(inputmeta_path),
            "--output-dir",
            str(infer_dir),
            "--dtype",
            "quantized",
            "--postprocess",
            "dump_results",
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"pegasus inference failed (code={proc.returncode})\n{proc.stdout[-4000:]}"
            )

        cand = sorted(
            glob.glob(str(infer_dir / f"iter_0_*_{NUM_CLASSES}_1_{NUM_FRAMES}.tensor"))
        )
        if not cand:
            cand = sorted(glob.glob(str(infer_dir / "iter_0_*out0*.tensor")))
        if not cand:
            raise FileNotFoundError(f"output tensor not found in {infer_dir}")

        vals = np.loadtxt(cand[0], dtype=np.float32)
        expect = 1 * NUM_CLASSES * 1 * NUM_FRAMES
        if vals.size != expect:
            raise ValueError(f"bad tensor size={vals.size}, expected={expect}, file={cand[0]}")
        y = vals.reshape(1, NUM_CLASSES, 1, NUM_FRAMES)
        return decode_logits_to_text(y)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

def whisper_inference(wav_path):
    x = wav_to_model_input(wav_path)
    if INFER_BACKEND == "onnx":
        y = onnx_sess.run(None, {"audio_signal": x})[0]  # [1,2049,1,38]
        return decode_logits_to_text(y)
    if INFER_BACKEND == "nb":
        return nb_inference_from_input(x)
    raise ValueError(f"unsupported INFER_BACKEND={INFER_BACKEND} (use onnx|nb)")

def remove_spaces(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", "", text)


def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def calc_cer(gt: str, pred: str) -> float:
    if metrics is not None:
        return metrics.get_cer(gt, pred)["cer"]
    ed = edit_distance(gt, pred)
    return ed / max(1, len(gt))
    
def natural_key(s):
    return [
        int(t) if t.isdigit() else t
        for t in re.split(r'(\d+)', os.path.basename(s))
    ]
    
def test_csv(test_file, save_dir):
    df = pd.read_csv(test_file)
    df = df[["FileName", "gt"]]
    results = []
    
    for idx, row in tqdm(df.iterrows()):
        file_path = row["FileName"]
        answer = remove_spaces(str(row["gt"]))

        # 파일 존재 확인
        if not os.path.exists(file_path):
            print(f"[WARN] 파일 없음: {file_path}")
            results.append([file_path, row["gt"], "", None, None])
            continue

        # 추론
        start_time = time.time()
        inf_text = whisper_inference(file_path)
        process_time = time.time() - start_time
        inf_no_space = remove_spaces(inf_text)

        # CER 계산
        cer = calc_cer(answer, inf_no_space)

        results.append([
            file_path,
            answer,
            inf_no_space,
            cer,
            cer * 100.0,
            process_time
        ])
    
    result_df = pd.DataFrame(
        results,
        columns=["FileName", "gt", "inf", "cer", "cer_percent", "process_time"],
    )
    avg_cer = result_df['cer'].mean()
    avg_cer_percent = avg_cer * 100.0
    avg_time = result_df['process_time'].mean()
    out_name = (
        f"{Path(test_file).stem}_{INFER_BACKEND}_result_"
        f"CER{avg_cer_percent:.2f}_TIME{avg_time:.2f}.csv"
    )
    result_df.to_csv(f"{save_dir}/{out_name}", index=False, encoding="utf-8-sig")
    
    print(f"BACKEND: {INFER_BACKEND}")
    print(f"AVG CER(%): {avg_cer_percent:.2f}")
    print(f"AVG CER(raw): {avg_cer:.5f}")
    print(f"AVG TIME: {avg_time}")
    
def test_dir(test_file):
    print("Input is a directory")

    results = []  # (filename, stt_text)
    audio_paths = []

    for root, _, files in os.walk(test_file):
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                audio_paths.append(os.path.join(root, f))

    audio_paths = sorted(audio_paths, key=natural_key)

    for audio_path in tqdm(audio_paths):
        inf_text = whisper_inference(audio_path)
        results.append((os.path.basename(audio_path), inf_text))

    # TXT 저장 (파일명 \t 텍스트)
    output_txt = "stt_texts_only.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        for _, stt_text in results:
            f.write(f"{stt_text}\n")

    print(f"Saved {len(results)} texts to {output_txt}")
    
def test_file(test_file):
    start_time = time.time()
    inf_text = whisper_inference(test_file)
    process_time = time.time() - start_time 
    print(f"[{INFER_BACKEND}] {test_file} processed for {process_time:.3f}")
    print(inf_text)


if __name__ == '__main__':
    save_dir = './whisper_test'
    
    test_csv_list = [
        '/nas04/nlp_sk/STT/data/test/007.저음질_eval_p.csv',
        '/nas04/nlp_sk/STT/data/test/009.한국어_강의_eval_p.csv',
        '/nas04/nlp_sk/STT/data/test/010.회의음성_eval_p.csv',
        '/nas04/nlp_sk/STT/data/test/012.상담음성_eval_p.csv',
        '/nas04/nlp_sk/STT/data/test/eval_clean_p.csv',
        '/nas04/nlp_sk/STT/data/test/eval_other_p.csv',
        '/nas04/nlp_sk/STT/data/test/modelhouse_2m_noheater.csv',
        '/nas04/nlp_sk/STT/data/test/modelhouse_2m.csv',
        '/nas04/nlp_sk/STT/data/test/modelhouse_3m.csv',
        '/nas04/nlp_sk/STT/data/test/7F_HJY.csv',
        '/nas04/nlp_sk/STT/data/test/7F_KSK.csv'
    ]
    
    for test_file in test_csv_list:
        print(f"[Start Processing] {test_file}")
        test_csv(test_file, save_dir)
        print(f"[Test Done] {test_file}")
        

# bin_u8_to_wav.py
import numpy as np
import soundfile as sf  # pip install soundfile

BIN_PATH = "input_0.bin"    # 네가 만든 u8 raw
OUT_WAV  = "restored.wav"
SCALE = 0.004333            # 로그의 scale
ZP    = 132                 # 로그의 zero_point
SR    = 16000               # 샘플레이트(너 것에 맞게 수정)

# 1) u8 읽기
q = np.fromfile(BIN_PATH, dtype=np.uint8)

# 2) dequantize → float32
x = (q.astype(np.float32) - ZP) * SCALE    # 대략 -0.572 ~ +0.533 범위

# 3) (선택) -1..1 클리핑 후 int16로 저장
x = np.clip(x, -1.0, 1.0)
x16 = (x * 32767.0).astype(np.int16)

# 4) WAV 쓰기 (모노)
sf.write(OUT_WAV, x16, SR, subtype="PCM_16")
print("WAV saved:", OUT_WAV)


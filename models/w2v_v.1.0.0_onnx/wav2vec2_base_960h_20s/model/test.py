import numpy as np
import soundfile as sf

audio, sr = sf.read('test.wav')
print(f"원본: {audio.shape} ({audio.shape[0]/16000:.1f}초)")

# 320000으로 패딩 (부족한 부분은 0으로 채움)
if len(audio) < 320000:
    padded = np.zeros(320000, dtype=np.float32)
    padded[:len(audio)] = audio
    audio = padded
    print(f"패딩 후: {audio.shape}")

# (1, 320000) 형태로
audio = audio.reshape(1, -1).astype(np.float32)
np.save('sample.npy', audio)
print(f"저장 완료: {audio.shape}")

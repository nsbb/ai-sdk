import librosa
import numpy as np

INPUT_WAV = "test.wav"
OUTPUT_NPY = "test.npy"
TARGET_LENGTH = 320000  # 샘플 개수 (20초)
TARGET_SR = 16000       # ✅ 샘플레이트 16kHz

# 16kHz로 리샘플링
data, sr = librosa.load(INPUT_WAV, sr=TARGET_SR)  # ✅ 이제 16000

current_length = data.shape[0]
print(f"변환 전 사이즈: {current_length} (약 {current_length/TARGET_SR:.2f}초)")

if current_length > TARGET_LENGTH:
    data = data[:TARGET_LENGTH]
    print("-> 너무 길어서 뒤를 잘랐습니다.")
elif current_length < TARGET_LENGTH:
    pad_width = TARGET_LENGTH - current_length
    data = np.pad(data, (0, pad_width), mode='constant', constant_values=0)
    print(f"-> 너무 짧아서 뒤에 0을 {pad_width}개 채웠습니다.")

data = data.reshape(1, -1)
print(f"최종 데이터 형태: {data.shape}")
np.save(OUTPUT_NPY, data)
print(f"저장 완료: {OUTPUT_NPY}")

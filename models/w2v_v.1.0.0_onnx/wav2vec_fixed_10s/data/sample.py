import librosa
import numpy as np

# 설정
INPUT_WAV = "sample.wav"
OUTPUT_NPY = "sample.npy"
TARGET_LENGTH = 160000  # 16kHz 기준 10초
TARGET_SR = 16000       # wav2vec2는 보통 16000Hz 필수

# 1. 오디오 로드 (자동으로 16000Hz로 리샘플링됨)
# data는 numpy array 형태입니다.
try:
    data, sr = librosa.load(INPUT_WAV, sr=TARGET_SR)
except Exception as e:
    print("librosa가 없으면: pip install librosa")
    exit()

# 2. 현재 사이즈 확인
current_length = data.shape[0]
print(f"변환 전 사이즈: {current_length} (약 {current_length/TARGET_SR:.2f}초)")

# 3. 자르기 또는 패딩 (160000으로 맞추기)
if current_length > TARGET_LENGTH:
    # [CASE 1] 너무 길면 -> 자르기
    data = data[:TARGET_LENGTH]
    print("-> 너무 길어서 뒤를 잘랐습니다.")
    
elif current_length < TARGET_LENGTH:
    # [CASE 2] 너무 짧으면 -> 패딩 (뒤에 0 채우기)
    pad_width = TARGET_LENGTH - current_length
    # (앞쪽패딩, 뒤쪽패딩)
    data = np.pad(data, (0, pad_width), mode='constant', constant_values=0)
    print(f"-> 너무 짧아서 뒤에 0을 {pad_width}개 채웠습니다.")
    
else:
    print("-> 이미 사이즈가 딱 맞습니다.")

# 4. 최종 사이즈 확인 및 배치 차원 추가
# 모델에 넣으려면 보통 (1, 160000) 형태여야 함
# 그냥 저장하고 싶으면 아래 reshape 줄은 지우셔도 됩니다.
data = data.reshape(1, -1) 

print(f"최종 데이터 형태(Shape): {data.shape}")

# 5. npy로 저장
np.save(OUTPUT_NPY, data)
print(f"저장 완료: {OUTPUT_NPY}")

import numpy as np
import os

# 출력 디렉토리
os.makedirs("data", exist_ok=True)

# 50개 샘플 생성
npy_paths = []
for idx in range(50):
    # [1, 80, 1, 300] 형태의 안전한 노이즈
    data = np.random.normal(0, 1.0, (1, 80, 1, 300)).astype(np.float32)
    data = np.clip(data, -5.0, 5.0)  # 안전 범위로 클리핑
    
    # 저장
    out_path = os.path.abspath(f"data/calib_{idx:03d}.npy")
    np.save(out_path, data)
    npy_paths.append(out_path)
    print(f"Generated {idx+1}/50")

# 리스트 파일 생성
with open("calib_dataset.txt", 'w') as f:
    for path in npy_paths:
        f.write(path + '\n')

print(f"✅ 완료: {len(npy_paths)}개 파일 생성")


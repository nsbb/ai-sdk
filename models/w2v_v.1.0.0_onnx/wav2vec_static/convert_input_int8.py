import numpy as np

# 1) npy 불러오기
x = np.load("sample.npy")  # shape: (36288,) or (36288,1) 등

# 2) wav2vec nb 로그의 양자화 파라미터
scale = 0.004333
zero_point = 132

# 3) 양자화 (asymmetric affine)
q = np.round(x / scale) + zero_point
q = np.clip(q, 0, 255).astype(np.uint8)

# 4) 네트워크가 기대하는 메모리 레이아웃에 맞게 reshape (필요시)
q = q.reshape(36288, 1)  # 로그의 dim에 맞게 조정

# 5) raw 바이너리로 저장 (header 없는 순수 바이트)
q.tofile("input_0.bin")

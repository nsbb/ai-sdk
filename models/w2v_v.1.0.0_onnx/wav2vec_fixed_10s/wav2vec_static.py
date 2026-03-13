# 파일명: fix_onnx_shape.py
import onnx
from onnx import shape_inference
from onnx import helper

# 1. 현재 ONNX 로드
model = onnx.load("wav2vec.onnx")

# 2. 입력 shape 확인
print("===== 현재 입력 =====")
for input_tensor in model.graph.input:
    print(f"Name: {input_tensor.name}")
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
             for d in input_tensor.type.tensor_type.shape.dim]
    print(f"Shape: {shape}")

# 3. 입력 shape를 고정 크기로 변경
for input_tensor in model.graph.input:
    # 첫 번째 입력만 수정 (보통 'input' 이름)
    dims = input_tensor.type.tensor_type.shape.dim
    dims[0].dim_value = 1        # batch = 1
    dims[1].dim_value = 160000   # time = 320000 (20초)

# 4. Shape inference 실행 (모든 중간 레이어 shape 재계산)
print("\n===== Shape inference 실행 중... =====")
try:
    model = shape_inference.infer_shapes(model)
    print("✅ Shape inference 성공!")
except Exception as e:
    print(f"⚠️ Shape inference 실패: {e}")
    print("그래도 저장은 해봅니다...")

# 5. 저장
output_path = "wav2vec_fixed_10s.onnx"
onnx.save(model, output_path)
print(f"\n✅ 저장 완료: {output_path}")

# 6. 확인
model_fixed = onnx.load(output_path)
print("\n===== 수정된 입력 =====")
for input_tensor in model_fixed.graph.input:
    shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
    print(f"Name: {input_tensor.name}, Shape: {shape}")

import onnx

# ==========================================
# 설정
# ==========================================
INPUT_MODEL = "wav2vec2_base_960h_20s.onnx"   # 원본 파일명
OUTPUT_MODEL = "wav2vec2_base_960h_5s.onnx"   # 저장할 파일명
NEW_LENGTH = 80000                            # 5초 (16000 * 5)

print(f"Loading {INPUT_MODEL}...")
model = onnx.load(INPUT_MODEL)

# ==========================================
# 1. 입력 사이즈 변경 (320000 -> 80000)
# ==========================================
# graph.input[0]이 보통 오디오 입력입니다.
input_tensor = model.graph.input[0]
old_length = input_tensor.type.tensor_type.shape.dim[1].dim_value

print(f"변경 전 입력 길이: {old_length}")

# 강제로 80000으로 값 변경
input_tensor.type.tensor_type.shape.dim[1].dim_value = NEW_LENGTH

print(f"변경 후 입력 길이: {NEW_LENGTH}")

# ==========================================
# 2. 중간 쉐이프 정보(value_info) 삭제 (★핵심★)
# ==========================================
# 이걸 안 지우면 NPU 툴이 "어? 입력은 5초인데 중간 기록은 20초네?" 하고 에러 냅니다.
# 싹 지워버리면 컴파일러가 입력 크기에 맞춰서 새로 계산합니다.
print("Cleaning up old shape info (value_info)...")
del model.graph.value_info[:]

# ==========================================
# 3. 출력 사이즈 초기화 (선택 사항)
# ==========================================
# 출력값의 길이도 입력에 따라 줄어들어야 하므로,
# 고정된 값(예: 999)을 지워주면 컴파일러가 알아서 추론하기 좋습니다.
# (혹시 에러나면 이 부분은 주석 처리하세요)
for output in model.graph.output:
    # 출력의 시간 차원(보통 dim[1])을 0(unknown)이나 -1로 만드는 대신
    # 그냥 기록된 차원을 비워버리는 게 안전할 수 있습니다.
    # 여기서는 간단히 두 번째 차원(Time step) 값을 지웁니다.
    if len(output.type.tensor_type.shape.dim) > 1:
        # dim_value를 0으로 만들면 보통 Dynamic으로 인식하거나 재계산합니다.
        output.type.tensor_type.shape.dim[1].dim_value = 0 

# ==========================================
# 4. 저장
# ==========================================
print(f"Saving to {OUTPUT_MODEL}...")
onnx.save(model, OUTPUT_MODEL)

print("✅ 완료! 이제 이 파일로 Pegasus 변환을 돌리세요.")

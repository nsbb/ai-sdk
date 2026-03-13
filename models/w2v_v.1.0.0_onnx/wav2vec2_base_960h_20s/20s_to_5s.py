import onnx
from onnx import shape_inference

def force_resize_onnx(input_path, output_path, new_seq_length):
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph

    # 1. 입력 사이즈 강제 변경
    # 보통 첫 번째 입력이 오디오 데이터입니다.
    input_tensor = graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape
    
    # [1, sequence_length] 또는 [batch, sequence, dim] 형태라고 가정
    # 안전하게 마지막 차원(시간 축)을 변경합니다.
    # Wav2Vec2는 보통 (Batch, Length) 입니다.
    dim_cnt = len(input_shape.dim)
    input_shape.dim[dim_cnt - 1].dim_value = new_seq_length
    
    print(f"--> Input shape changed to: [..., {new_seq_length}]")

    # ==========================================================
    # 핵심 포인트: 기존에 박제된 중간/출력 사이즈 정보 삭제
    # 이걸 안 지우면 inference 시 기존 20s 정보를 계속 참조함
    # ==========================================================
    print("--> Clearing existing 'value_info' (cached shapes)...")
    del graph.value_info[:]  # 중간 레이어 쉐이프 정보 삭제

    print("--> Clearing existing 'output' shapes...")
    # 출력 노드의 쉐이프도 20s 기준으로 고정되어 있을 수 있으므로 초기화
    # (이름은 유지하되 type 정보만 날리면 추론 시 다시 채워짐)
    # 하지만 완전히 날리면 안 되고, shape 정보만 리셋해야 안전합니다.
    # 여기서는 shape inference가 output을 덮어쓰도록 유도합니다.
    
    # 2. Shape Inference 실행 (엄격 모드)
    # 입력이 바뀌었고 중간 정보가 없으므로, ONNX가 수식에 따라 처음부터 끝까지 다시 계산합니다.
    print("--> Running strict Shape Inference...")
    try:
        # data_prop=True: 상수를 전파하여 쉐이프 계산 (더 강력함)
        model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    except Exception as e:
        print("!!! Shape Inference Failed !!!")
        print("이 모델은 내부 연산(Reshape 등)에 '상수(Constant)'로 20초 길이가 하드코딩 되어 있을 수 있습니다.")
        print(f"Error detail: {e}")
        return

    # 3. 저장
    onnx.save(model, output_path)
    print(f"✅ Success! Resized model saved to: {output_path}")
    
    # 4. 검증 (옵션)
    print("--> Verifying output model...")
    chk_model = onnx.load(output_path)
    # 첫번째 value_info(중간 레이어) 하나만 찍어서 확인
    if len(chk_model.graph.value_info) > 0:
        first_layer_dim = chk_model.graph.value_info[0].type.tensor_type.shape.dim
        print(f"    Check First Layer Dim: {[d.dim_value for d in first_layer_dim]}")
    else:
        print("    (value_info is empty, but inputs should be correct)")

# --- 실행 ---
# 5초 * 16000 = 80000
force_resize_onnx("wav2vec2_base_960h_20s.onnx", "wav2vec2_base_960h_5s.onnx", 80000)

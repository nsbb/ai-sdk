import torch
from transformers import Wav2Vec2ForCTC

def export_model():
    model_id = "facebook/wav2vec2-base-960h" 
    print(f"Loading model: {model_id}...")

    # ==============================================================================
    # 핵심 수정 사항: attn_implementation="eager"
    # 이 옵션을 켜면 최신 PyTorch의 SDPA 커널 대신, 고전적인 MatMul/Softmax 방식을 사용합니다.
    # 이렇게 해야 ONNX Opset 12에서도 문제없이 변환됩니다.
    # ==============================================================================
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_id, attn_implementation="eager")
    except TypeError:
        # 혹시 transformers 버전이 낮아서 이 옵션을 모를 경우를 대비한 예외처리
        print("Warning: 'attn_implementation' not supported, falling back to default.")
        model = Wav2Vec2ForCTC.from_pretrained(model_id)

    model.eval()

    # 16000Hz * 5s = 80000 samples
    target_seq_length = 80000
    dummy_input = torch.randn(1, target_seq_length, requires_grad=False)

    output_onnx_path = "wav2vec2_base_960h_5s_static.onnx"

    print(f"Exporting ONNX model to {output_onnx_path} ...")
    print(f"Input Shape: [1, {target_seq_length}] (Opset 12)")

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=12,          # NPU 호환성을 위해 12 유지
        do_constant_folding=True,
        input_names=['input_values'],
        output_names=['logits'],
        
        # NPU(RK3588/T527)용이므로 dynamic_axes는 끄는 것을 권장합니다.
        # 필요하다면 아래 주석 해제
        # dynamic_axes={'input_values': {1: 'sequence_length'}, 'logits': {1: 'sequence_length'}}
    )

    print(f"✅ Success! Saved to {output_onnx_path}")

if __name__ == "__main__":
    export_model()

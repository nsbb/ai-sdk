from transformers import Wav2Vec2ForCTC
import torch

print("1. 모델 로딩...")
model = Wav2Vec2ForCTC.from_pretrained(".", local_files_only=True)
model.eval()

print("2. 5초(80000) 입력으로 export...")
torch.onnx.export(
    model,
    torch.randn(1, 80000),  # 5초
    "wav2vec_5s.onnx",
    input_names=['input'],
    output_names=['logits'],
    opset_version=11,
    do_constant_folding=True
)

print("3. 확인...")
import onnx
m = onnx.load("wav2vec_5s.onnx")
for inp in m.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  Shape: {dims}")
print("✅ wav2vec_5s.onnx 생성 완료!")

from transformers import Wav2Vec2Model
import torch
import onnx

print("1. 로컬 모델 로딩...")
model = Wav2Vec2Model.from_pretrained(".", local_files_only=True)
model.eval()

print("2. 10초(160000) 입력으로 export...")
dummy = torch.randn(1, 160000)

torch.onnx.export(
    model,
    dummy,
    "wav2vec_clean_10s.onnx",
    input_names=['input'],
    output_names=['logits'],
    opset_version=11,
    do_constant_folding=True
)

print("3. 확인...")
m = onnx.load("wav2vec_clean_10s.onnx")
for inp in m.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  Shape: {dims}")
print("✅ 완료!")

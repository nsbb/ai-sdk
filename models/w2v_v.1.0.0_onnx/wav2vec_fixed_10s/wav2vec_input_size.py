import onnxruntime as ort
path = 'wav2vec_5s.onnx'
session = ort.InferenceSession(path)

for input_meta in session.get_inputs():
    print(f"Name: {input_meta.name}")
    print(f"Shape: {input_meta.shape}")
    print(f"Type: {input_meta.type}")


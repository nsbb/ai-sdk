import onnxruntime as ort
#path = 'wav2vec_fixed_10s.onnx'
path = 'wav2vec2_base_960h_5s.onnx'
session = ort.InferenceSession(path)

for input_meta in session.get_inputs():
    print(f"Name: {input_meta.name}")
    print(f"Shape: {input_meta.shape}")
    print(f"Type: {input_meta.type}")

for output_meta in session.get_outputs():
    print(f"Name: {output_meta.name}")
    print(f"Shape: {output_meta.shape}")
    print(f"Type: {output_meta.type}")

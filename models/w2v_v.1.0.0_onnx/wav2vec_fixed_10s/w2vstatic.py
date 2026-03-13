import onnx

model = onnx.load("wav2vec.onnx")
input_tensor = model.graph.input[0]

input_tensor.type.tensor_type.shape.dim[0].dim_value = 1
input_tensor.type.tensor_type.shape.dim[1].dim_value = 160000

onnx.save(model,"wav2vec_static.onnx")


import onnx

model = onnx.load("citrinet_npu.onnx")

for input in model.graph.input:
    print(input.name)
    for dim in input.type.tensor_type.shape.dim:
        print(dim.dim_value, dim.dim_param)

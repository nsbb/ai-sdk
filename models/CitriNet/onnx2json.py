import onnx
from google.protobuf.json_format import MessageToJson

model = onnx.load("citrinet_npu.onnx")
json_str = MessageToJson(model)

with open("citrinet_npu.json", "w") as f:
    f.write(json_str)


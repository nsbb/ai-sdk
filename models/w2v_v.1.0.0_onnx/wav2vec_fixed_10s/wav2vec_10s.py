import onnx
from onnx import numpy_helper

model = onnx.load("wav2vec_clean_10s.onnx")

# 모든 노드의 attribute 확인 및 수정
for node in model.graph.node:
    for attr in node.attribute:
        # Constant 노드의 값 수정
        if attr.name == "value" and attr.HasField('t'):
            tensor = attr.t
            arr = numpy_helper.to_array(tensor)
            
            # 320000 관련 상수 찾아서 160000으로 변경
            if 320000 in arr or 63999 in arr or 31999 in arr:
                print(f"Node: {node.name}, 발견: {arr}")
                # 여기서 수정...

onnx.save(model, "wav2vec_truly_fixed_10s.onnx")

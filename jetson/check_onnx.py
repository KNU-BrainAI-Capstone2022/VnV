import sys
import onnx 
import onnx_graphsurgeon as gs

file_name = '../checkpoint/deeplabv3plus_mobilenet_cityscapes/model_best_jetson.onnx'
model = onnx.load(file_name)

onnx.checker.check_model(model,full_check=True)
onnx.checker.check_graph(model.graph)

print(f"onnx.__version__={onnx.__version__!r}, opset={onnx.defs.onnx_opset_version()}, IR_VERSION={onnx.IR_VERSION}")

graph = gs.import_onnx(model)

for node in graph.nodes:
    if 'Pad' in node.name:
        print(node)
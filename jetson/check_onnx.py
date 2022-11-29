import sys
import onnx 
import onnx_graphsurgeon as gs
import tensorrt as trt
import pycuda
import numpy as np
import onnxruntime
file_name = '../checkpoint/deeplabv3plus_mobilenet_cityscapes/model_best_jetson.onnx'
model = onnx.load(file_name)

onnx.checker.check_model(model,full_check=True)
onnx.checker.check_graph(model.graph)

print(f"onnx.__version__ = {onnx.__version__!r}, opset={onnx.defs.onnx_opset_version()}, IR_VERSION={onnx.IR_VERSION}")
print(f"onnxruntime.__version__ = {onnxruntime.__version__}")
print(f"onnx_graphsurgeon.__version__ = {gs.__version__}")
print(f"tensorrt.__version__ = {trt.__version__}")
print(f"pycuda.VERSION = {pycuda.VERSION}")
print(f"numpy.__version__ = {np.__version__}")

# onnx check
graph = gs.import_onnx(model)

for node in graph.nodes:
    if 'Pad' in node.name:
        print(node)

# trt check
with trt.Logger() as logger, trt.Runtime(logger) as runtime:
    print(f"logger = {logger}")
    print(f"runtime = {runtime}")
    context = engine.create_execution_context()
    print(context)
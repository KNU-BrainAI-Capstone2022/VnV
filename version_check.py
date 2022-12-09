import sys
import onnx 
import onnx_graphsurgeon as gs
import tensorrt as trt
import pycuda
import numpy as np
import onnxruntime
import argparse
import torch
import torchvision
import os
def get_args():

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Video Encoding")

    parser.add_argument("-c","--checkpoint", type=str, help="model checkpoint path")

    return parser.parse_args()

if __name__=="__main__":
    kargs = vars(get_args())
    if os.path.exists(kargs["checkpoint"]):
        file_name = kargs["checkpoint"]
        model = onnx.load(file_name)

        onnx.checker.check_model(model,full_check=True)
        onnx.checker.check_graph(model.graph)

    print(f"torch.__version__ = {torch.__version__}")
    print(f"torchvision.__version__ = {torchvision.__version__}")
    print(f"onnx.__version__ = {onnx.__version__!r}, opset={onnx.defs.onnx_opset_version()}, IR_VERSION={onnx.IR_VERSION}")
    print(f"onnxruntime.__version__ = {onnxruntime.__version__}")
    print(f"onnx_graphsurgeon.__version__ = {gs.__version__}")
    print(f"tensorrt.__version__ = {trt.__version__}")
    print(f"pycuda.VERSION = {pycuda.VERSION}")
    print(f"numpy.__version__ = {np.__version__}")


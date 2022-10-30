import cv2
import time
import models
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
from utils import Dataset,Util
import argparse
import torch_tensorrt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnxruntime

# Read trt file
def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # Deserialization engine
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# This operation is a general function
def infer(context, input_img, output_size, batch_size):
    # Convert input data to Float32. If this type needs to be converted, there will be many errors
    input_img = input_img.astype(np.float32)
    # Create output array to receive data
    output = np.empty(output_size, dtype=np.float32)

    # Allocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.nbytes)
    d_output = cuda.mem_alloc(batch_size * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.execute_async(batch_size, bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    stream.synchronize()

    # Return predictions
    return output

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Video Encoding")
    # model option
    parser.add_argument("--weights", type=str, default=None,help="model weights path")
    # Dataset Options
    parser.add_argument("--input", type=str, help="input video name",required=True)
    parser.add_argument("--pair", action='store_true', help="Generate pair frame")
    parser.add_argument("--test", action='store_true', help="Generate thunbnail")
    parser.add_argument("--fp16", action='store_true', help="data type fp16")
    parser.add_argument("--int8", action='store_true', help="data type int8")
    return parser.parse_args()
    
if __name__=='__main__':
    # model load``
    kargs=vars(get_args())
    kargs['weights']= "checkpoint/deeplabv3plus_resnet50_cityscapes/model_best.onnx"
    print(f"\nWeights loading...\n")
    # ckpt=os.path.join(ckpt_dir,kargs['model']+'_'+kargs['type']+'.trt')
    if not os.path.exists(kargs['weights']):
        print('Weights is not exist\n')
        exit(1)

    # load pytorch model
    onnx_file_path = "checkpoint/deeplabv3plus_resnet50_cityscapes/model_best.onnx"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(f'device {device}')

    # video write
    input_video = 'video/'+kargs['input']+'.mp4'
    if not os.path.exists('./video'):
        os.mkdir('./video')
    if not os.path.exists(input_video):
        print('input video is not exist\n')
        exit(1)
    cap = cv2.VideoCapture(input_video)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'video ({frame_width},{frame_height}), {fps} fps')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if kargs['pair']:
        out_name = 'video/'+kargs['input']+'_output_pair.mp4'
        out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,2*frame_height))
    else:
        out_name = 'video/'+kargs['input']+'_output.mp4'
        out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,frame_height))
    print(f'{input_video} encoding ...')

    # cmap load
    classes = Dataset.CustomCityscapesSegmentation('dataset')
    cmap = classes.getcmap()

    # # Read trt file
    # engine = loadEngine2TensorRT(kargs['weights'])
    # # Create context
    # context = engine.create_execution_context()
    print(f'\nonnx weights loading ...')
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
    binding = session.io_binding()

    total_frame=0
    with torch.no_grad():
        start = time.time()
        print("Start TensorRT Test...")
        while total_frame<10:
            ret, frame = cap.read()
            if not ret:
                print('cap.read is failed')
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            predict = F.to_tensor(frame).unsqueeze(0).to(device, dtype=torch.float32)
            predict = F.normalize(predict,(0.485,0.456,0.406),(0.229,0.224,0.225))

            # compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(predict)}
            ort_outs = ort_session.run(None, ort_inputs)
            img_out_y = ort_outs[0]

            img_out_y = np.squeeze(img_out_y,axis=0)
            img_out_y = np.argmax(img_out_y,axis=0)
            result = Util.mask_colorize(img_out_y,cmap)
            result = cv2.addWeighted(frame,0.3,result,0.7,0)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
            if kargs['pair']:
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                result = cv2.vconcat([frame,result])
            if kargs['test']:
                cv2.imwrite('video/test.jpg',result)
                print('Generate test.jpg')
                exit(1)
            else:
                out_cap.write(result)
            total_frame +=1
    print(f'finish encoding - {out_name}')
    total_time = time.time()-start
    print(f'total frame = {total_frame} \ntotal time = {total_time:.2f}s')
    print(f'average time = {total_time/total_frame:.2f}')
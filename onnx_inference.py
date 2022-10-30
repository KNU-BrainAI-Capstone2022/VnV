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
    parser.add_argument("--video", type=str, help="input video name",required=True)
    parser.add_argument("--pair", action='store_true', help="Generate pair frame")
    parser.add_argument("--test", action='store_true', help="Generate thunbnail")
    parser.add_argument("--fp16", action='store_true', help="data type fp16")
    parser.add_argument("--int8", action='store_true', help="data type int8")
    return parser.parse_args()
    
if __name__=='__main__':
    # model load``
    kargs=vars(get_args())
    # fp32
    # kargs['weights']= "checkpoint/deeplabv3plus_resnet50_cityscapes/model_best.onnx"
    
    # fp16
    kargs['weights'] = "test_half.onnx"
    
    print(f"\nWeights loading...\n")
    # ckpt=os.path.join(ckpt_dir,kargs['model']+'_'+kargs['type']+'.trt')
    if not os.path.exists(kargs['weights']):
        print('Weights is not exist\n')
        exit(1)

    # load pytorch model
    onnx_file_path = kargs['weights']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(f'device {device}')

    # video write
    input_video = 'video/'+kargs['video']+'.mp4'
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
        out_name = 'video/'+kargs['video']+'_output_pair.mp4'
        out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,2*frame_height))
    else:
        out_name = 'video/'+kargs['video']+'_output.mp4'
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
    # # slow than pytorch
    session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # onnx_opt = onnxruntime.SessionOptions()
    # onnx_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # onnx_opt.enable_profiling=True
    # session = onnxruntime.InferenceSession(onnx_file_path, onnx_opt, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    binding = session.io_binding()
    
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    
    total_frame=0
    
    start = time.time()
    print("Start onnx Test...")
    while total_frame<30:
        # print(total_frame)
        ret, frame = cap.read()
        if not ret:
            print('cap.read is failed')
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # inputs = frame / 255.0
        # inputs = np.transpose(inputs, (2,0,1))
        # inputs = (inputs- mean.reshape(-1,1,1)) / std.reshape(-1,1,1)
        # # print(inputs.shape)
        # inputs = np.expand_dims(inputs,axis=0).astype(np.float32)
        # # print(inputs.shape, inputs.dtype)

        # # input cuda
        # inputs = onnxruntime.OrtValue.ortvalue_from_numpy(inputs,device_type='cuda',device_id=0)

        # binding.bind_cpu_input('input',inputs)
        # binding.bind_output('outputs')
        # session.run_with_iobinding(binding)
        # img_out_y = binding.copy_outputs_to_cpu()[0]
        # # print(img_out_y.shape, type(img_out_y))

        # # -------------------------------------------
        # # for input, output -> torch.to(device)
        # # -------------------------------------------
        # inputs = F.to_tensor(frame).unsqueeze(0)
        # inputs = F.normalize(inputs,(0.485,0.456,0.406),(0.229,0.224,0.225))
        # inputs = inputs.contiguous().to(device,dtype=torch.float16)
        
        # binding.bind_input(
        #     name='inputs',
        #     device_type='cuda',
        #     device_id=0,
        #     element_type=np.float32,
        #     shape=tuple(inputs.shape),
        #     buffer_ptr=inputs.data_ptr(),
        #     )
        # ## Allocate the PyTorch tensor for the model output
        # outputs = torch.empty((1,19,1080,1920), dtype=torch.float16,device='cuda:0').contiguous()
        
        # binding.bind_output(
        #     name='outputs',
        #     device_type='cuda',
        #     device_id=0,
        #     element_type=np.float32,
        #     shape=tuple(outputs.shape),
        #     buffer_ptr=outputs.data_ptr(),
        # )

        # session.run_with_iobinding(binding)

        # # ort_output=OrtValue
        # ort_output = binding.get_outputs()[0]
        # # ort_output = binding.copy_outputs_to_cpu()
        # img_out_y = ort_output.numpy()

        # # last
        # inputs = F.to_tensor(frame).unsqueeze(0)
        # inputs = F.normalize(inputs,(0.485,0.456,0.406),(0.229,0.224,0.225))
        # inputs = inputs.contiguous()
        
        # # inputs = onnxruntime.OrtValue.ortvalue_from_numpy(inputs)
        # img_out_y = session.run(["outputs"], {"inputs": to_numpy(inputs)})[0]

        # # -------------------------------------
        # # input output is cpu
        # # -------------------------------------
        inputs = frame / 255.0
        inputs = np.transpose(inputs, (2,0,1))
        inputs = (inputs- mean.reshape(-1,1,1)) / std.reshape(-1,1,1)
        # print(inputs.shape)
        inputs = np.expand_dims(inputs,axis=0).astype(np.float16)
        # print(inputs.shape, inputs.dtype)
        ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(inputs)
        img_out_y = session.run(["outputs"], {"inputs": ortvalue})[0]
        # print(img_out_y.shape, type(img_out_y))

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
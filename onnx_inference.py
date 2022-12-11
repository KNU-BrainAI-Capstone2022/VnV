# onnx == 1.11.0
# onnxruntime == 1.10.0
# http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/tutorial_onnxruntime/inference.html 

import cv2
import time
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import argparse
import onnx
import onnxruntime
from utils.colormap import mask_colorize,cmap_cityscapes,cmap_voc


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Video Encoding")
    # model option
    parser.add_argument('-c', "--checkpoint", type=str, default=None,help="model weights path")
    # Dataset Options
    parser.add_argument("--test", action='store_true', help="Generate thunbnail")
    parser.add_argument("--int8", action='store_true', help="data type int8")
    parser.add_argument("--fp16", action='store_true', help="data type fp16")
    return parser.parse_args()
    
if __name__=='__main__':
    # model load``
    kargs = vars(get_args())

    # fp16
    kargs['weights'] = "./checkpoint/deeplabv3_mobilenetv3_voc2012_plain.onnx"
    
    print(f"\nWeights loading...\n")
    if not os.path.exists(kargs['weights']):
        print('Weights is not exist\n')
        exit(1)

    # load pytorch model
    onnx_file_path = kargs['weights']
    device = onnxruntime.get_device()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(f'device {device}')

 
    frame_width = 640
    frame_height = 360
    img_name = "test.jpg"
    out_name = 'test_output.jpg'

    # cmap load
    cmap = np.array(cmap_voc,dtype=np.uint8)

    print(f'\nonnx weights loading ...')

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    # # slow than pytorch
    session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # onnx_opt = onnxruntime.SessionOptions()
    # # onnx_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # onnx_opt.enable_profiling=True
    # session = onnxruntime.InferenceSession(onnx_file_path, onnx_opt, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    print(f"input name : {session.get_inputs()[0].name}")
    print(f"input shape : {session.get_inputs()[0].shape}")
    print(f"input type : {session.get_inputs()[0].type}")
    
    print(f"output name : {session.get_outputs()[0].name}")
    print(f"output shape : {session.get_outputs()[0].shape}")
    print(f"output type : {session.get_outputs()[0].type}")
    
    binding = session.io_binding()
    total_frame=1
    
    
    em = cv2.imread("test.jpg")
    ori_img = cv2.resize(em, (frame_width,frame_height))
    em = cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    em = np.asarray(em).astype('float32')/255.0
    em =np.moveaxis(em, 2, 0)
    em = np.expand_dims(em,axis=0).astype(np.float32)
    print(em.shape, em.dtype)
    # inputs = onnxruntime.OrtValue.ortvalue_from_numpy(inputs,device_type='cuda',device_id=0)
    # input cpu
    # inputs = onnxruntime.OrtValue.ortvalue_from_numpy(em)
    
    # print(f"equal test -> {np.array_equal(inputs.numpy(), em)}")
    # print(f"inputs.device_name() : {inputs.device_name()}")  # 'cpu'
    # print(f"inputs.shape() : {inputs.shape()}")   # shape of the numpy array X
    # print(f"inputs.data_type() : {inputs.data_type()}")  # 'cpu'
    # print(f"inputs.is_tensor() : {inputs.is_tensor()}")  #  # 'tensor(float)'
    # result = session.run(["outputs"], {"inputs": inputs})[0][0]
    # print(result.shape, type(result))
    
    binding.bind_cpu_input('inputs',em)
    binding.bind_output('outputs')
    session.run_with_iobinding(binding)
    result = binding.copy_outputs_to_cpu()[0][0][0]
    print(result.shape, result.dtype)
    
    # result = np.squeeze(result,axis=0)
    # result = np.argmax(result,axis=0)
    result = mask_colorize(result,cmap)
    # print(result.shape)
    result = cv2.addWeighted(ori_img,0.3,result,0.7,0)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # cv2.imwrite('test_output.jpg',result)
    # print('Generate test.jpg')
    
    exit(1)
    
    start = time.time()
    print("Start onnx Test...")
    while total_frame<30:
        # print(total_frame)
        ret, frame = cap.read()
        if not ret:
            print('cap.read is failed')
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        inputs = frame / 255.0
        inputs = np.transpose(inputs, (2,0,1))
        # inputs = (inputs- mean.reshape(-1,1,1)) / std.reshape(-1,1,1)
        # print(inputs.shape)
        inputs = np.expand_dims(inputs,axis=0).astype(np.float32)
        # # print(inputs.shape, inputs.dtype)

        # # input cuda
        inputs = onnxruntime.OrtValue.ortvalue_from_numpy(inputs,device_type='cuda',device_id=0)

        binding.bind_cpu_input('input',inputs)
        binding.bind_output('outputs')
        session.run_with_iobinding(binding)
        img_out_y = binding.copy_outputs_to_cpu()[0]
        print(img_out_y.shape, type(img_out_y))

        # # -------------------------------------------
        # # for input, output -> torch.to(device)
        # # -------------------------------------------
        # inputs = F.to_tensor(frame).unsqueeze(0)
        # inputs = F.normalize(inputs,(0.485,0.456,0.406),(0.229,0.224,0.225))
        # inputs = inputs.contiguous().to(device,dtype=torch.float32)
        
        # binding.bind_input(
        #     name='inputs',
        #     device_type='cuda',
        #     device_id=0,
        #     element_type=np.float32,
        #     shape=tuple(inputs.shape),
        #     buffer_ptr=inputs.data_ptr(),
        #     )
        # ## Allocate the PyTorch tensor for the model output
        # outputs = torch.empty((1,19,1080,1920), dtype=torch.float32,device='cuda:0').contiguous()
        
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

        # # -------------------------------------
        # # last
        # # -------------------------------------
        # inputs = F.to_tensor(frame).unsqueeze(0)
        # inputs = F.normalize(inputs,(0.485,0.456,0.406),(0.229,0.224,0.225))
        # inputs = inputs.contiguous()
        
        # # inputs = onnxruntime.OrtValue.ortvalue_from_numpy(inputs)
        # img_out_y = session.run(["outputs"], {"inputs": to_numpy(inputs)})[0]

        # # -------------------------------------
        # # input output is cpu
        # # -------------------------------------
        # inputs = frame / 255.0
        # inputs = np.transpose(inputs, (2,0,1))
        # inputs = (inputs- mean.reshape(-1,1,1)) / std.reshape(-1,1,1)
        # # print(inputs.shape)
        # inputs = np.expand_dims(inputs,axis=0).astype(np.float16)
        # # print(inputs.shape, inputs.dtype)
        # ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(inputs)
        # img_out_y = session.run(["outputs"], {"inputs": ortvalue})[0]
        # # print(img_out_y.shape, type(img_out_y))

        img_out_y = np.squeeze(img_out_y,axis=0)
        img_out_y = np.argmax(img_out_y,axis=0)
        result = Util.mask_colorize(img_out_y,cmap)
        result = cv2.addWeighted(frame,0.3,result,0.7,0)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        cv2.imwrite('video/test.jpg',result)
        print('Generate test.jpg')
        exit(1)

print(f'finish encoding - {out_name}')
total_time = time.time()-start
print(f'total frame = {total_frame} \ntotal time = {total_time:.2f}s')
print(f'average time = {total_time/total_frame:.2f}')
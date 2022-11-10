import torch
import numpy as np
import io
import os
from torch import nn
import cv2
import argparse
import time

from utils import Dataset,Util
import models

import torch.onnx
import torchvision

import pycuda.driver 
import pycuda.autoinit
from torch2trt import torch2trt, TRTModule
from utils import Dataset,Util
import onnx
import onnxruntime
import torchvision.transforms.functional as F

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__=='__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: 19, cityscapes)")
    available_models = sorted(name for name in models.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              models.model.__dict__[name])
                              )
    parser.add_argument("--model", choices=available_models, default="deeplabv3plus_resnet50", type=str, help="model name")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16]) # DeepLab Only
    parser.add_argument("--weights", type=str,default='./checkpoint/deeplabv3plus_resnet50_cityscapes/model_best.pth', help='weight file path')
    parser.add_argument("--fp16", action='store_true', help='Create tensorrt fp16')
    parser.add_argument("--int8", action='store_true', help='Create tensorrt int8')
    kargs = vars(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.model.__dict__[kargs['model']](num_classes=kargs['num_classes'],output_stride=16,pretrained_backbone=False)

    # class map
    classes = Dataset.CustomCityscapesSegmentation('dataset')
    cmap = classes.getcmap()
    
    # load weight
    dict_model = torch.load(kargs['weights'])
    model.load_state_dict(dict_model['model_state'])

    # test image load
    img = cv2.imread('./video/test.jpg')
    ori_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = F.to_tensor(ori_img).unsqueeze(0).to(device,torch.float16)
    img = F.normalize(img,(0.485,0.456,0.406),(0.229,0.224,0.225))
    
    # # ---------------------------------
    # # convert torch2trt
    # # ---------------------------------
    # model = model.to(device)
    # model.eval()
    # input_size = torch.randn(1,3,1080,1920)
    # trt_model = torch2trt(model,[input_size], max_workspace_size=1<<34,fp16_mode=True)
  
    # print(f'\ntorch2trt testing ....')
    # # # inputs = torch.from_numpy(img)
    # output = trt_model(inputs)
    # print(f'output shape = {output.shape}, {type(output)}')

    # print(output.device)
    # output = output.detach()///
    # output = Util.mask_colorize(output,cmap).astype(np.uint8)
    
    # result = cv2.addWeighted(img,0.3,output,0.7,0)
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('video/torch2trt-test.jpg', result)

    # del y, y_trt, input_size, model_trt, model, inputs, output

    # # ------------------------------------------------
    # # create onnx file
    # # ------------------------------------------------
    # model=model.to(device).half()
    # model.eval()
    # input_size = torch.randn(1,3,1080,1920).to(device, dtype=torch.float16)
    # torch_out = model(input_size)
    
    # print(f'\nCreating onnx file...')
    # torch.onnx.export(
    #     model,                      # 모델
    #     input_size,                 # 모델 입력값
    #     'test_half.onnx',                  # 모델 저장 경로
    #     verbose=True,              # 변환 과정
    #     export_params=True,         # 모델 파일 안에 학습된 모델 가중치 저장
    #     opset_version = 11,         # onnx 버전
    #     input_names=['inputs'],      # 모델의 입력값을 가리키는 이름
    #     output_names= ['outputs'],   # 모델의 아웃풋 이름
    # )
    # print(f'Converting onnx file is done')

    #print(model)
    # ------------------------------------------
    # onnx model load 
    # ------------------------------------------
    print(f'\nonnx weights loading ...')
    onnx_model = onnx.load("test.onnx")
    onnx.checker.check_model(onnx_model)

    session = onnxruntime.InferenceSession('test.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    binding = session.io_binding()
    start = time.time()

    # # ----------------------------------------
    # # for gpu input, output
    # # ----------------------------------------
    # # inputs = img.contiguous()

    # inputs = img.contiguous().to(device)

    # binding.bind_input(
    #     name='inputs',
    #     device_type='cuda',
    #     device_id=0,
    #     element_type=np.float32,
    #     shape=tuple(inputs.shape),
    #     buffer_ptr=inputs.data_ptr(),
    #     )
    # ## Allocate the PyTorch tensor for the model output
    # outputs = torch.empty((1,19,1080,1920), dtype=torch.float32, device='cuda:0').contiguous()
    
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
    # img_out_y = ort_output.numpy()
    # print(img_out_y.shape, type(img_out_y))

    # -----------------------------------------------------
    # compute ONNX Runtime output prediction for cpu
    # -----------------------------------------------------
    inputs = {session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = session.run(None, inputs)
    img_out_y = ort_outs[0]
    print(img_out_y.shape)

    img_out_y = np.squeeze(img_out_y,axis=0)
    img_out_y = np.argmax(img_out_y,axis=0)
    result = Util.mask_colorize(img_out_y,cmap)
    print(result.shape, type(result))
    result = cv2.addWeighted(ori_img,0.3,result,0.7,0)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('video/onnx-test-fp16.jpg', result)
    print(f'onnx time : {time.time()-start}')
    
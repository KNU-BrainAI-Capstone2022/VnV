import torch
import os
import cv2
import argparse

import models
import torch.onnx

import pycuda.driver 
import pycuda.autoinit
from torch2trt import torch2trt, TRTModule

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: 19, cityscapes)")
    available_models = sorted(name for name in models.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              models.model.__dict__[name])
                              )
    parser.add_argument("--model", choices=available_models, default="deeplabv3plus_mobilenet", type=str, help="model name")
    parser.add_argument("--weights", type=str,default='./checkpoint/deeplabv3plus_mobilenet_cityscapes/model_best.pth', help='weight file path')
    parser.add_argument("--onnx", action='store_true', help='Create onnx fp32')
    parser.add_argument("--trt", action='store_true', help='Create tensorrt model')
    parser.add_argument("--onnx-ver", type=int, default=8, help='Onnx version')
    kargs = vars(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.model.__dict__[kargs['model']](
        num_classes=kargs['num_classes'],output_stride=kargs['output_stride'],pretrained_backbone=False
        )
    
    # load weight
    dict_model = torch.load(kargs['weights'])
    model.load_state_dict(dict_model['model_state'])

    # cityscape image size
    frame_width = 2048
    frame_height = 1024
    model = model.eval()
    model = model.to(device)
    model = model.half()
    
    input_size = torch.randn(1,3,frame_height,frame_width,requires_grad=True).to(device)
    #print(model)
    # torch --> onnx
    if kargs['onnx']:
        save_name = f"{kargs['weights'][:-4]}_jetson.onnx"
        print(f'\nCreating onnx file...')
        torch.onnx.export(
            model,                      # 모델
            input_size,                 # 모델 입력값
            save_name,                  # 모델 저장 경로
            verbose=True,              # 변환 과정
            export_params=True,         # 모델 파일 안에 학습된 모델 가중치 저장
            opset_version = kargs['onnx-ver'],         # onnx 버전
            input_names=['inputs'],      # 모델의 입력값을 가리키는 이름
            output_names= ['outputs'],   # 모델의 아웃풋 이름
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )
        print(f"{kargs['model']}.pth -> onnx is done")

    # onnx - > tensorrt
    # /usr/src/tensorrt/bin/trtexec --onnx= model.onnx --saveEngine=model.trt
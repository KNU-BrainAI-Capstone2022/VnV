# torch == 1.9.0
# onnx == 1.8.1
# onnx.ai.opeset == 13
# onnxruntime == 1.7.0 

import sys
import os
sys.path.append("../")
import torch
import argparse
from models.model import deeplabv3plus_resnet50,deeplabv3plus_mobilenet
import torch.onnx
from torch2trt import torch2trt
from utils_jet.model import WrappedModel, TestModel

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: 19, cityscapes)")
    parser.add_argument("--weights", type=str,default='../checkpoint/deeplabv3plus_mobilenet_cityscapes/model_best.pth', help='weight file path')
    parser.add_argument("--onnx", action='store_true', help='Create onnx fp32')
    parser.add_argument("--trt", action='store_true', help='Create tensorrt model')
    parser.add_argument("--onnx-opset", type=int, default=13, help='Opset version ai.onnx')
    parser.add_argument("-O","--output", type=str, default=None)
    parser.add_argument("--int8", action='store_true', help="Create torch2trt int8")
    parser.add_argument("--fp16", action='store_true', help="Create torch2trt fp16")
    parser.add_argument("--fp32", action='store_true', help="Create torch2trt fp32")

    kargs = vars(parser.parse_args())
    print(f'args : {kargs}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device : {device}')

    if 'resnet' in kargs['weights']:
        model = deeplabv3plus_resnet50(num_classes=kargs['num_classes'],pretrained_backbone=False)
    elif 'mobilenet' in kargs['weights']:
        model = deeplabv3plus_mobilenet(num_classes=kargs['num_classes'], pretrained_backbone=False)
    
    if kargs['output'] is None:
        output_name = kargs['weights'].replace(".pth","_jetson")
    else:
        output_name = kargs['output']

    # load weight
    print(f'Load model....')
    model_state= torch.load(kargs['weights'])
    model.load_state_dict(model_state['model_state'])
    del model_state

    # wrapping
    # model = WrappedModel(model)
    
    # Test
    input_shapes = (1,3,270,480)
    # model = TestModel(kargs['num_classes'],input_shapes)
    
    # cityscape image size
    model.eval()
    model = model.half().cuda()
    
    input_size = torch.randn(input_shapes,dtype=torch.half)

    print(f'input shape : {input_size.shape} ({input_size.dtype})')

    # torch --> onnx
    if kargs['onnx']:

        save_name = output_name + '.onnx'

        print(f'\nCreating onnx file...')
        torch.onnx.export(
            model,                      # 모델
            input_size,                 # 모델 입력값
            save_name,                  # 모델 저장 경로
            verbose=True,              # 변환 과정
            export_params=True,         # 모델 파일 안에 학습된 모델 가중치 저장
            opset_version = kargs['onnx_opset'],         # onnx 버전
            input_names=['inputs'],      # 모델의 입력값을 가리키는 이름
            output_names= ['outputs'],   # 모델의 아웃풋 이름
            operator_export_type = torch.onnx.OperatorExportTypes.ONNX
        )
        print(f"{save_name} -> onnx is done")

    # onnx - > tensorrt
    # /usr/src/tensorrt/bin/trtexec --onnx=model_best_jetson.onnx --saveEngine=model_best_jetson_fp16.engine --fp16 --verbose --buildOnly

    # torch -> tensorrt 
    if kargs['trt']:
        input_size = input_size.cuda()
        #out = model(input_size)
        if kargs['int8']:
            print(f'\nCreating trt int8 file...')
            trt_model = torch2trt(model,[input_size], max_workspace_size=1<<32,int8_mode=True,use_onnx=True,onnx_opset=kargs['onnx_opset'])

            torch.save(trt_model.state_dict(),f"{output_name}_trt_int8.pth")
            print(f"\nTRTModule {output_name}_trt_int8.pth is Created")

        if kargs['fp16']:
            print(f'\nCreating trt fp16 file...')
            trt_model = torch2trt(model,[input_size], max_workspace_size=1<<32,fp16_mode=True,use_onnx=True,onnx_opset=kargs['onnx_opset'])

            torch.save(trt_model.state_dict(),f"{output_name}_trt_fp16.pth")
            print(f"\nTRTModule {output_name}_trt_fp16.pth is Created")
        
        if kargs['fp32']:
            print(f'\nCreating trt float32 file...')
            trt_model = torch2trt(model,[input_size], max_workspace_size=1<<32,use_onnx=True, onnx_opset=kargs['onnx_opset'])

            torch.save(trt_model.state_dict(),f"{output_name}_trt_fp32.pth")
            print(f"\nTRTModule {output_name}_trt_fp32.pth is Created")



        

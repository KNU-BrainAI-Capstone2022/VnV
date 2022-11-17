import torch
import argparse
import os
from ..models.model import deeplabv3plus_resnet50,deeplabv3plus_mobilenet
import torch.onnx
from torch2trt import torch2trt
import torchvision

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: 19, cityscapes)")
    parser.add_argument("--weights", type=str,default='../checkpoint/deeplabv3plus_resnet50_cityscapes/model_best.pth', help='weight file path')
    parser.add_argument("--onnx", action='store_true', help='Create onnx fp32')
    parser.add_argument("--trt", action='store_true', help='Create tensorrt model')
    parser.add_argument("--onnx-ver", type=int, default=13, help='Opset version ai.onnx')

    kargs = vars(parser.parse_args())
    print(f'args : {kargs}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device : {device}')
    if 'resnet' in kargs['weights']:
        model = deeplabv3plus_resnet50(num_classes=19,pretrained_backbone=False)
    elif 'mobilenet' in kargs['weights']:
        model = deeplabv3plus_mobilenet(num_classes=kargs['num_classes'], pretrained_backbone=False)
    # load weight
    print(f'Load model....')
    model_state= torch.load(kargs['weights'])
    model.load_state_dict(model_state['model_state'])
    del model_state

    # cityscape image size
    model.eval()
    model = model.cuda().half()

    input_size = torch.randn((1,3,540,960),dtype=torch.half,device=device)
    print(f'input shape : {input_size.shape} ({input_size.dtype})')
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
            opset_version = kargs['onnx_ver'],         # onnx 버전
            input_names=['inputs'],      # 모델의 입력값을 가리키는 이름
            output_names= ['outputs'],   # 모델의 아웃풋 이름
            operator_export_type = torch.onnx.OperatorExportTypes.ONNX
        )
        print(f"{kargs['weights']}_jetson.pth -> onnx is done")

    # onnx - > tensorrt
    # /usr/src/tensorrt/bin/trtexec --onnx=model_best_jetson.onnx --saveEngine=model_best_jetson_fp16.engine --fp16 --verbose
    if kargs['trt']:
        print(f'\nCreating trt fp16 file...')
        trt_model = torch2trt(model,[input_size], max_workspace_size=1<<25,fp16_mode=True,use_onnx=True)
        torch.save(trt_model.state_dict(),f"{kargs['weights'][:-4]}_cityscapes_trt_fp16.pth")
        print(f"\nTRTModule {kargs['weights'][:-4]}_cityscapes_trt_fp16.pth is Created")


        

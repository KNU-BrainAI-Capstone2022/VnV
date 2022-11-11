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
    parser.add_argument("--model", choices=available_models, default="deeplabv3plus_resnet50", type=str, help="model name")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16]) # DeepLab Only
    parser.add_argument("--weights", type=str,default='./checkpoint/deeplabv3plus_resnet50_cityscapes/model_best.pth', help='weight file path')
    parser.add_argument("--video", type=str, default=None,help="input video name in video floder")
    parser.add_argument("--fp16", action='store_true', help='Create tensorrt fp16')
    parser.add_argument("--int8", action='store_true', help='Create tensorrt int8')
    parser.add_argument("--fp32", action='store_true', help='Create tensorrt fp32')
    parser.add_argument("--jit", action='store_true', help='Create pytorch jit')
    parser.add_argument("--onnx", action='store_true', help='Create onnx fp32')
    parser.add_argument("--trt", action='store_true', help='Create tensorrt model')
    parser.add_argument("--onnx-ver", type=int, default=11, help='Onnx version')
    kargs = vars(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.model.__dict__[kargs['model']](num_classes=kargs['num_classes'],output_stride=kargs['output_stride'],pretrained_backbone=False).to(device)
    
    # load weight
    dict_model = torch.load(kargs['weights'])
    model.load_state_dict(dict_model['model_state'])

    # video write
    if kargs['video']:
        input_video = 'video/'+kargs['video']+'.mp4'
        if not os.path.exists('./video'):
            os.mkdir('./video')
        if not os.path.exists(input_video):
            print('input video is not exist\n')
            exit(1)
        print(f'input_video = {input_video}')
        # print cap info
        # print(cv2.getBuildInformation())
        cap = cv2.VideoCapture(input_video)
        if cap.isOpened():
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'video ({frame_width},{frame_height}), {fps} fps')
        else:
            print(f'video is not opened')
    else:
        # cityscape image size
        frame_width = 2048
        frame_height = 1024
    model.eval()

    input_size = torch.randn(1,3,frame_height,frame_width,requires_grad=True).to(device,torch.float32)
    torch_out = model(input_size)
    #print(model)
    # torch --> onnx
    if kargs['onnx']:
        save_name = f"{kargs['weights'][:-4]}.onnx"
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

    # pytorch -> tensorrt
    if kargs['trt']:
        if kargs['int8']:
            print(f'\nCreating trt int8 file...')
            trt_model = torch2trt(model,[input_size], max_workspace_size=1<<32,int8_mode=True,use_onnx=True)
            if kargs['video']:
                torch.save(trt_model.state_dict(),f"{kargs['weights'][:-4]}_trt_int8.pth")
                print(f"\nTRTModule {kargs['weights'][:-4]}_trt_int8.pth is Created")
            else:
                torch.save(trt_model.state_dict(),f"{kargs['weights'][:-4]}_cityscapes_trt_int8.pth")
                print(f"\nTRTModule {kargs['weights'][:-4]}_cityscapes_trt_int8.pth is Created")
        if kargs['fp32']:
            print(f'\nCreating trt float32 file...')
            trt_model = torch2trt(model,[input_size], max_workspace_size=1<<32,use_onnx=True)
            if kargs['video']:
                torch.save(trt_model.state_dict(),f"{kargs['weights'][:-4]}_trt_fp32.pth")
                print(f"\nTRTModule {kargs['weights'][:-4]}_trt_float32.pth is Created")
            else:
                torch.save(trt_model.state_dict(),f"{kargs['weights'][:-4]}_cityscapes_trt_fp32.pth")
                print(f"\nTRTModule {kargs['weights'][:-4]}_cityscapes_trt_float32.pth is Created")
        if kargs['fp16']:
            print(f'\nCreating trt fp16 file...')
            trt_model = torch2trt(model,[input_size], max_workspace_size=1<<32,fp16_mode=True,use_onnx=True)
            if kargs['video']:
                torch.save(trt_model.state_dict(),f"{kargs['weights'][:-4]}_trt_fp16.pth")
                print(f"\nTRTModule {kargs['weights'][:-4]}_trt_fp16.pth is Created")
            else:
                torch.save(trt_model.state_dict(),f"{kargs['weights'][:-4]}_cityscapes_trt_fp16.pth")
                print(f"\nTRTModule {kargs['weights'][:-4]}_cityscapes_trt_fp16.pth is Created")
            

    # Torchscript module 저장
    
    if kargs['jit']:
        try: 
            print(f'\nCreating jit file...')
            script_model = torch.jit.script(model)
            script_model.save(f"{kargs['weights'][:-4]}.ts")
            print(f"Jit script {kargs['weights'][:-4]}.ts is Created")
        except Exception as e:
            print(e)
            print('unable to convert jit script')
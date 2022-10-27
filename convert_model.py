import torch
import numpy as numpy
import io
import os

from torch import nn

import torch.onnx
import models
import argparse
import cv2

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: None)")
    available_models = sorted(name for name in models.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              models.model.__dict__[name])
                              )
    parser.add_argument("--model", choices=available_models, default="deeplabv3plus_resnet50", type=str, help="model name")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16]) # DeepLab Only
    parser.add_argument("--weights", type=str,default='./checkpoint/deeplabv3plus_resnet50_cityscapes/model_best.pth', help='weight file')
    parser.add_argument("--video", type=str, help="input video name in video floder",required=True)
    kargs = vars(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.model.__dict__[kargs['model']](num_classes=kargs['num_classes'],output_stride=16,pretrained_backbone=False).to(device)
    
    # load weight
    dict_model = torch.load(kargs['weights'])
    model.load_state_dict(dict_model['model_state'])

    # video write
    input_video = 'video/'+kargs['video']+'.mp4'
    if not os.path.exists('./video'):
        os.mkdir('./video')
    if not os.path.exists(input_video):
        print('input video is not exist\n')
        exit(1)
    print(f'input_video = {input_video}')
    print(cv2.getBuildInformation())
    cap = cv2.VideoCapture(input_video)
    if cap.isOpened():
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'video ({frame_width},{frame_height}), {fps} fps')
    else:
        print(f'video is not opened')


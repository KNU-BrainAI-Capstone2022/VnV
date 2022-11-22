import sys
sys.path.append('../')
import torch
import time
import numpy as np
import models
import torchvision.transforms.functional as F
import cv2
from torch2trt import TRTModule
import argparse
from utils import Util
from utils.Dataset import CustomCityscapesSegmentation
import os

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Video Encoding")
    # model option
    parser.add_argument("--weights", type=str, default=None,help="model weights path")
    parser.add_argument("--dtype", type=str, choices=['fp32','fp16','int8'], default='fp32',help="weight dtype")
    # Dataset Options
    parser.add_argument("--video", type=str, help="input video name",required=True)
    return parser.parse_args()

if __name__=='__main__':
    # model load``
    kargs=vars(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cmap load
    cmap = CustomCityscapesSegmentation.cmap

    print(cmap)
    # --------------------------------------------
    # load model
    # --------------------------------------------
    if kargs['weights']:
        model_path = kargs['weights']
    else:
        model_path = f"checkpoint/deeplabv3plus_resnet50_cityscapes/model_best_trt_{kargs['dtype']}.pth"
    if not os.path.exists(model_path):
        print('model is not exist\n')
        exit(1)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))
    
    # model dtype
    if kargs['dtype']=='int8':
        model_dtype = torch.qint8
    else:
        model_dtype = torch.float32

    # --------------------------------------------
    # video info check
    # --------------------------------------------
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

    # ----------------------------------------------
    # video write
    # ----------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_name = 'video/'+kargs['video']+'_'+kargs['dtype']+'_output.mp4'
    out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,frame_height))
    print(f'{input_video} encoding ...')

    # ----------------------------------------------
    # video encoding
    # ----------------------------------------------
    start = time.time()
    total_frame =0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('cap.read is failed')
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        predict = F.to_tensor(frame).unsqueeze(0).to(device, dtype=model_dtype).half()
        predict = F.normalize(predict,(0.485,0.456,0.406),(0.229,0.224,0.225))
        
        # model inference
        predict = model_trt(predict)
    
        predict = predict.detach().argmax(dim=1).squeeze(0).cpu().numpy()
        predict = Util.mask_colorize(predict,cmap).astype(np.uint8)
        
        result = cv2.addWeighted(frame,0.3,predict,0.7,0)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        out_cap.write(result)
        total_frame +=1

    print(f'finish encoding - {out_name}')
    total_time = time.time()-start
    print(f'total frame = {total_frame} \ntotal time = {total_time:.2f}s')
    print(f'average time = {total_time/total_frame:.2f}')

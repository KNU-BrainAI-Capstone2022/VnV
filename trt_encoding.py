import torch
import time
import numpy as np
import models
import torchvision.transforms.functional as F
import cv2
from torch2trt import TRTModule
import argparse
from utils import Util
import os

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Video Encoding")
    # model option
    parser.add_argument("--weights", type=str, default=None,help="model weights path")
    parser.add_argument("--dtype", type=str, choices=['fp32','fp16','int8'], default='fp32',help="weight dtype")
    # Dataset Options
    parser.add_argument("--video", type=str, help="input video name",required=True)
    return parser.parse_args()

def getcmap():
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    ]
if __name__=='__main__':
    # model load``
    kargs=vars(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cmap load
    cmap = classes.getcmap()

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
    out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,2*frame_height))
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

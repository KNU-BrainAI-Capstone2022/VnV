import cv2
import time
import models
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
from utils import Dataset,Util
import argparse


def getcmap():
    # cityscapes cmap
    train_cmap = [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)
    ]
    return train_cmap

def get_args():

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Video Encoding")
    # model option
    parser.add_argument("--model", type=str, help="model name",required=True)
    # Dataset Options
    parser.add_argument("--video", type=str, help="input video name",required=True)
    parser.add_argument("--pair", action='store_true', help="Generate pair frame")
    parser.add_argument("--test", action='store_true', help="Generate thunbnail")

    return parser.parse_args()
    
if __name__=="__main__":
    # model load
    kargs=vars(get_args())

    print(f"\n'{kargs['model']}' model loading...\n")
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # ckpt_dir = os.path.join(root_dir,"checkpoint",kargs['model']+'_cityscapes')
    # ckpt=os.path.join(ckpt_dir,'model_best.pth')
    # if not os.path.exists(ckpt):
    #     print('model is not exist\n')
    #     exit(1)
    kargs['model'] = 'deeplabv3plus_mobilenet'
    ckpt = './checkpoint/deeplabv3plus_mobilenet_new_distill_0.1_0.01_fix_cityscapes/model_best.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.model.__dict__[kargs['model']](num_classes=19,output_stride=16,pretrained_backbone=False).to(device)

    dict_model = torch.load(ckpt)
    model.load_state_dict(dict_model['model_state'])

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
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'video ({frame_width},{frame_height}), {fps} fps')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if kargs['pair']:
        out_name = 'video/'+kargs['video']+'_'+kargs['model']+'_output_pair.mp4'
        out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,2*frame_height))
    else:
        out_name = 'video/'+kargs['video']+'_'+kargs['model']+'_output.mp4'
        out_cap = cv2.VideoWriter(out_name,fourcc,fps,(frame_width,frame_height))

    print(f'{input_video} encoding ...')

    # cmap load
    cmap = getcmap()

    model.eval()
    total_frame=0
    only_infer_time = 0
    with torch.no_grad():
        start = time.time()
        while True:
            if total_frame > 30:
                break
            ret, frame = cap.read()
            if not ret:
                print('cap.read is failed')
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            predict = F.to_tensor(frame).unsqueeze(0).to(device, dtype=torch.float32)
            only_run = time.time()
            predict = model(predict)
            only_infer_time += time.time()-only_run
            predict = predict.detach().argmax(dim=1).squeeze(0).cpu().numpy()
            predict = Util.mask_colorize(predict,cmap).astype(np.uint8)
            
            result = cv2.addWeighted(frame,0.3,predict,0.7,0)
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
    print(f'average time = {total_time/total_frame:.2f}s')
    print(f'Only inference time : {only_infer_time:.2f}s')
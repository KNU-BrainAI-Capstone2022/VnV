import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# 모델 저장 함수
def save(ckpt_dir,model,optim,epoch,best_val=None,filename="model_epoch"):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if filename == "model_epoch":
        filepath = os.path.join(ckpt_dir,f"model_epoch{epoch}.pth")
    else:
        filepath = os.path.join(ckpt_dir,filename)
    torch.save({'model':model.state_dict(),
                'optim':optim.state_dict(),
                'epoch':epoch,
                'best_val':best_val},filepath)


# 모델 로드 함수
def load(ckpt_dir,model,optim,name):
    ckpt = os.path.join(ckpt_dir,name)
    if not os.path.exists(ckpt):
        epoch = 0
        best_val = 0
        print("There is no checkpoint")
        return model,optim,epoch,best_val

    dict_model = torch.load(ckpt)

    model.load_state_dict(dict_model['model'])
    optim.load_state_dict(dict_model['optim'])
    epoch = dict_model['epoch']
    best_val = dict_model['best_val']
    return model,optim,epoch,best_val

def voc_denorm(x,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)):
    for i in range(x.shape[0]):
        x[i] = (x[i]* std[i]) + mean[i]
    return x

def mask_colorize(masks,colormap):
    color_map = torch.tensor(colormap)
    r_mask = torch.zeros_like(masks,dtype=torch.uint8)
    g_mask = torch.zeros_like(masks,dtype=torch.uint8)
    b_mask = torch.zeros_like(masks,dtype=torch.uint8)
    for k in range(len(colormap)):
        indices = masks == k
        r_mask[indices] = color_map[k,0]
        g_mask[indices] = color_map[k,1]
        b_mask[indices] = color_map[k,2]
    return torch.cat([r_mask,g_mask,b_mask],dim=1)

def make_figure(images,outputs,targets,colormap):
    n=images.size(0)
    fig, ax = plt.subplots(3,1,figsize=(n*3,9))
    ax[0].imshow(torchvision.utils.make_grid(images.cpu(), normalize=True).permute(1,2,0))
    ax[0].set_title("Input")
    ax[0].axis('off')
    targets = mask_colorize(targets,colormap)
    ax[1].imshow(torchvision.utils.make_grid(targets.cpu(), normalize=False).permute(1,2,0))
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')
    outputs = mask_colorize(outputs,colormap)
    ax[2].imshow(torchvision.utils.make_grid(outputs.cpu(), normalize=False).permute(1,2,0))
    ax[2].set_title("Prediction")
    ax[2].axis('off')
    fig.tight_layout()
    return fig

def make_iou_bar(iou,classes):
    fig = plt.figure(figsize=(len(classes),len(classes)//3))
    plt.bar(range(len(classes)),iou,width=0.5,tick_label=classes)
    fig.suptitle("IOU per Class")
    return fig

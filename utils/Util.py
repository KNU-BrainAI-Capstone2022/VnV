import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# 모델 저장 함수
def save(ckpt_dir,model,optim,epoch,best_val=0,time,filename="model_epoch"):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if filename == "model_epoch":
        filepath = os.path.join(ckpt_dir,f"model_epoch{epoch}.pth")
    else:
        filepath = os.path.join(ckpt_dir,filename)
    torch.save({'model':model.state_dict(),
                'optim':optim.state_dict(),
                'epoch':epoch,
                'best_val':best_val,
                'time':time},filepath)

# 모델 로드 함수
def load(ckpt_dir,name,model,optim):
    ckpt = os.path.join(ckpt_dir,name)
    if not os.path.exists(ckpt):
        epoch = 0
        best_val = 0
        time = 0
        print("There is no checkpoint")
        return model,optim,epoch,best_val,time

    dict_model = torch.load(ckpt)

    model.load_state_dict(dict_model['model'])
    optim.load_state_dict(dict_model['optim'])
    epoch = dict_model['epoch']
    best_val = dict_model['best_val']
    time = dict_model['time']
    return model,optim,epoch,best_val,time

def mask_colorize(masks,cmap):
    # masks : BxCxHxW
    # if C != 1, argmax
    if masks.size(1) == len(cmap):
        _, masks = masks.max(dim=1)
        masks = masks.unsqueeze(dim=1)
    cmap_ = torch.tensor(cmap)
    r_mask = torch.zeros_like(masks,dtype=torch.uint8)
    g_mask = torch.zeros_like(masks,dtype=torch.uint8)
    b_mask = torch.zeros_like(masks,dtype=torch.uint8)
    for k in range(len(cmap)):
        indices = masks == k
        r_mask[indices] = cmap_[k,0]
        g_mask[indices] = cmap_[k,1]
        b_mask[indices] = cmap_[k,2]
    return torch.cat([r_mask,g_mask,b_mask],dim=1)

def make_figure(images,targets,outputs,colormap):
    if targets.dim() == 3: # BxHxW
        targets = targets.unsqueeze(1)
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

def intersection_union(output:torch.Tensor,target:torch.Tensor,c:int,ignore_index=255):
    # output shape : BxCxHxW float
    # Target shape : BxHxW long
    assert output.dim() == 4
    assert target.dim() == 3

    output = torch.nn.functional.softmax(output,dim=1)
    _,output = torch.max(output,dim=1) # BxCxHxW -> BxHxW

    output = output.contiguous().view(-1).type(torch.float)
    target = target.contiguous().view(-1).type(torch.float)
    intersection = output[output==target]
    # Ignore Background
    area_intersection = torch.histc(intersection,bins=c,min=0,max=c-1)
    area_output = torch.histc(output,bins=c,min=0,max=c-1)
    area_target = torch.histc(target,bins=c,min=0,max=c-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.detach().cpu().numpy(),area_union.detach().cpu().numpy()

def make_iou_bar(iou,classes):
    fig = plt.figure(figsize=(len(classes),len(classes)//3))
    plt.bar(range(len(classes)),iou,width=0.5,tick_label=classes)
    fig.suptitle("IOU per Class")
    return fig

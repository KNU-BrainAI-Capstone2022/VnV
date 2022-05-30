#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50,deeplabv3_resnet101

from utils.Util import make_figure,make_iou_bar
from utils.Dataset import get_dataset
from utils.Transform import get_transform
from utils.Metric import intersection_union

# parser = argparse.ArgumentParser()
# parser.add_argument('--backbone',choices=['resnet50','resnet101'],required=True)
# args = vars(parser.parse_args())
# args = parser.parse_args()

def get_model(backbone:str):
    if backbone == 'resnet50':
        model = deeplabv3_resnet50(pretrained=True)
    elif backbone == 'resnet101':
        model = deeplabv3_resnet101(pretrained=True)
    return model

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(root_dir),"dataset","VOCdevkit","VOC2012")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_ds,num_classes = get_dataset(data_dir,'voc',"val",transform=get_transform(train=False,base_size=512))
data_loader = DataLoader(test_ds,batch_size=8,shuffle=False)
colormap = test_ds.colormap
classes = test_ds.classes

model = get_model('resnet50').to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
model.eval()
loss_arr=[]
total_intersection = np.zeros((num_classes,))
total_union = np.zeros((num_classes,))
with torch.no_grad():
    for data in data_loader:
        inputs, targets = data['input'].to(device), data['target'].to(device)
        # Forward
        outputs = model(inputs)['out']
        # Metric
        loss = criterion(outputs,targets.long())
        loss_arr.append(loss.item())
        intersection, union = intersection_union(outputs,targets,num_classes)
        total_intersection += intersection
        total_union += union

        loss_mean = np.mean(loss_arr)
        miou = np.nanmean(total_intersection) / np.nanmean(total_union)
        print(f"TEST: LOSS {loss_mean:.4f} | mIOU {miou:.4f}")
iou = total_intersection / total_union
miou = np.nanmean(total_intersection) / np.nanmean(total_union)
print(f"TEST: LOSS {loss_mean:.4f} | mIOU {miou:.4f}")
# figure
data = next(iter(data_loader))
inputs, targets = data['input'].to(device), data['target'].to(device)
outputs = model(inputs)['out']
fig = make_figure(inputs.detach().cpu(),targets.detach().cpu(),outputs.detach().cpu(),colormap)
iou_bar = make_iou_bar(np.nan_to_num(iou[1:]),classes[1:])
plt.show()
# %%

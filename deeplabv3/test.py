#%%
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50,deeplabv3_resnet101

parser = argparse.ArgumentParser()
parser.add_argument('--backbone',choices=['resnet50','resnet101'],required=True)
args = vars(parser.parse_args())
args = parser.parse_args()

def get_model(backbone:str):
    if backbone == 'resnet50':
        model = deeplabv3_resnet50(pretrained=True)
    elif backbone == 'resnet101':
        model = deeplabv3_resnet101(pretrained=True)
    return model

root_dir = '.'
data_dir = os.path.join(root_dir,"dataset","VOCdevkit","VOC2012")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_ds,num_classes = get_dataset(data_dir,args.dataset,"val",transform=get_transform(train=False,base_size=512))
test_loader = DataLoader(test_ds,batch_size=8,shuffle=False)

model = get_model(args.backbone).to(device)
model.eval()
loss_arr=[]
total_intersection = np.zeros((num_classes,))
total_union = np.zeros((num_classes,))
with torch.no_grad():
    for data in test_loader:
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
iou = total_intersection / total_union
miou = np.nanmean(total_intersection) / np.nanmean(total_union)
plt.subplot(2,1,1)
fig = make_figure(inputs,targets,outputs,colormap)
plt.subplot(2,1,2)
iou_bar = make_iou_bar(np.nanmean(iou_arr,axis=0),classes[1:])
print(f"TEST: LOSS {loss_mean:.4f} | mIOU {miou:.4f}")
plt.show()
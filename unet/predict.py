import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.Dataset import Dataset
from utils.Transform import ToTensor,Normalization
from models.model import Unet
from models.utils import *

# 경로
root_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(root_dir,'dataset','test')
ckpt_dir = os.path.join(root_dir,'checkpoint')
result_dir = os.path.join(root_dir,'result')
os.makedirs(result_dir,exist_ok=True)

# Parameter
batch_size = 3

# Transform/Dataloader
transform = transforms.Compose([Normalization(mean=0.5,std=0.5),
                                ToTensor()])
test_dataset = Dataset(test_dir,transform=transform)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

# 모델
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Unet().to(device)

# 손실 함수
loss_fn = torch.nn.BCEWithLogitsLoss()
# 옵티마이저
optim = torch.optim.Adam(net.parameters())

# Variable
num_data_test = len(test_dataset)
num_batch_test = int(np.ceil(num_data_test/batch_size))
# Function
fn_tonumpy = lambda x:x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x:(x*0.5)+0.5
fn_class = lambda x:1.0*(x>0.5)
# 모델 로드
net,optim,start_epoch = load(ckpt_dir,net=net,optim=optim)

with torch.no_grad():
    net.eval()
    loss_arr = []
    for batch, data in enumerate(test_loader,start=1):
        X,Y = data[0].to(device),data[1].to(device)
        pred = net(X)
        
        loss = loss_fn(pred,Y)
        loss_arr.append(loss.item())
        
        print(f"TEST: BATCH {batch:04d} / {num_batch_test:04d} | LOSS {np.mean(loss_arr):.4f}")
        
        input_ = fn_tonumpy(fn_denorm(X))
        label_ = fn_tonumpy(Y)
        output_ = fn_tonumpy(fn_class(pred))
        
        for i in range(input_.shape[0]):
            id = batch_size*(batch-1) + i
            plt.imsave(os.path.join(result_dir,f"input_{i:03d}.png"),input_[i].squeeze(),cmap="gray")
            plt.imsave(os.path.join(result_dir,f"label_{i:03d}.png"),label_[i].squeeze(),cmap="gray")
            plt.imsave(os.path.join(result_dir,f"output_{i:03d}.png"),output_[i].squeeze(),cmap="gray")

print(f"AVERAGE TEST: BATCH {batch:04d} / {num_batch_test:04d} | LOSS {np.mean(loss_arr):.4f}")
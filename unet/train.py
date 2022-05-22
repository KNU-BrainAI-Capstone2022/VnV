import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils.Dataset import Dataset
from utils.Transform import ToTensor,RandomFlip,Normalization
from models.model import Unet
from models.utils import *

# 경로 변수 선언
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir,"dataset")

train_dir = os.path.join(data_dir,"train")
val_dir = os.path.join(data_dir,"val")
ckpt_dir = os.path.join(root_dir,"checkpoint")
log_dir = os.path.join(root_dir,"logs")

# 학습에 필요한 Parameters
batch_size = 3
lr = 1e-3
num_epoch = 20
# Transform / DataLoader
transform = transforms.Compose([Normalization(mean=0.5,std=0.5),RandomFlip(),ToTensor()])

train_dataset = Dataset(data_dir=train_dir,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

val_dataset = Dataset(data_dir=val_dir,transform=transform)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

# 모델 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = Unet().to(device)
# 손실 함수 정의
loss_fn = torch.nn.BCEWithLogitsLoss()
# 옵티마이저 정의
optim = torch.optim.Adam(unet.parameters(),lr=lr)
# 부수적인 Variable
num_data_train = len(train_dataset)
num_data_val = len(val_dataset)

num_batch_train = int(np.ceil(num_data_train/batch_size))
num_batch_val = int(np.ceil(num_data_val/batch_size))
# 부수적인 Function
fn_tonumpy = lambda x:x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x,mean,std:(x*std)+mean
fn_class = lambda x:1.0 * (x > 0.5)
# Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,"train"))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir,"val"))
# 학습하던 모델 있으면 로드
net, optim, start_epoch = load(ckpt_dir=ckpt_dir,net=unet,optim=optim)

for epoch in range(start_epoch+1,num_epoch+1):
    # Train mode
    unet.train()
    loss_arr = []

    for batch, data in enumerate(train_loader,start=1):
        X,Y = data[0].to(device),data[1].to(device)
        # Forwardprop
        pred = unet(X)
        # Backprop
        optim.zero_grad()
        loss = loss_fn(pred,Y)
        loss.backward()
        optim.step()

        loss_arr.append(loss.item())

        print(f"TRAIN: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_train:04d} | LOSS {np.mean(loss_arr):.4f}")
        
        # Tensorboard
        input_ = fn_tonumpy(fn_denorm(X,mean=0.5,std=0.5))
        label_ = fn_tonumpy(Y)
        output_ = fn_tonumpy(fn_class(pred))
        
        writer_train.add_image('input',input_,num_batch_train*(epoch-1)+batch,dataformats='NHWC')
        writer_train.add_image('label',label_,num_batch_train*(epoch-1)+batch,dataformats='NHWC')
        writer_train.add_image('output',output_,num_batch_train*(epoch-1)+batch,dataformats='NHWC')
    
    writer_train.add_scalar('loss',np.mean(loss_arr),epoch)
    
    with torch.no_grad():
        unet.eval()
        loss_arr=[]
        
        for batch, data in enumerate(val_loader,start=1):
            X,Y = data[0].to(device),data[1].to(device)
            # Forwardprop
            pred = unet(X)
            # Loss 계산
            loss = loss_fn(pred,Y)
            loss_arr.append(loss.item())
            
            print(f"VALID: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_val:04d} | LOSS {np.mean(loss_arr):.4f}")
            
            input_ = fn_tonumpy(fn_denorm(X,mean=0.5,std=0.5))
            label_ = fn_tonumpy(Y)
            output_ = fn_tonumpy(fn_class(pred))
            
            writer_val.add_image('input',input_,num_batch_val*(epoch-1)+batch,dataformats='NHWC')
            writer_val.add_image('label',label_,num_batch_val*(epoch-1)+batch,dataformats='NHWC')
            writer_val.add_image('output',output_,num_batch_val*(epoch-1)+batch,dataformats='NHWC')
        
        writer_val.add_scalar('loss',np.mean(loss_arr),epoch)

    # epoch 5마다 저장
    if epoch % 5 == 0:
        save(ckpt_dir=ckpt_dir,net=unet,optim=optim,epoch=epoch)

writer_train.close()
writer_val.close()

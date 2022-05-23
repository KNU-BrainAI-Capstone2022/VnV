import os
import time
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader

from model import *
from utils import save,load
from transforms import transform_train,transform_eval

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--data_path", default="voc2012", type=str, help="dataset path")
    parser.add_argument("--dataset", default="voc", type=str, help="dataset name")
    parser.add_argument("--model", default="unet", type=str, help="model name")
    parser.add_argument("-j", "--num_workers", default=4, type=int, help="number of data loading workers (default: 16)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu")
    parser.add_argument("--epochs", default=30, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight-decay",default=1e-4,type=float,help="weight decay (default: 1e-4)",help="weight_decay",)
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--test-only",dest="test_only",help="Only test the model",action="store_true")
    return parser.parse_args()

def get_dataset(dir_path,name,image_set,transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)
    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
    }
    p, ds_fn, num_classes = paths[name]
    ds = ds_fn(p, image_set=image_set,download=True,transform=transform)
    return ds,num_classes

# 부수적인 Function
fn_tonumpy = lambda x:x.to('cpu').detach().numpy().transpose(0,2,3,1)
def fn_denorm(x,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)):
    for i in range(x.shape[0]):
        x[i] = (x[i]* std[i]) + mean[i]
    return x

def train_one_epoch(model,criterion,optimizer,data_loader,lr_scheduler,device,epoch):
    model.train()
    loss_arr = []
    for batch, (input, target) in enumerate(data_loader,1):
        input, target = input.to(device), target.to(device)
        # Forward
        output = model(input)
        # Backward
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_arr.append(loss.item())
        # Result
        print(f"TRAIN: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_train:04d} | LOSS {np.mean(loss_arr):.4f}")
        # Tensorboard
        input_ = fn_tonumpy(fn_denorm(X,mean=0.5,std=0.5))
        label_ = fn_tonumpy(Y)
        output_ = fn_tonumpy(fn_class(pred))

        writer_train.add_image('input',input_,num_batch_train*(epoch-1)+batch,dataformats='NHWC')
        writer_train.add_image('label',label_,num_batch_train*(epoch-1)+batch,dataformats='NHWC')
        writer_train.add_image('output',output_,num_batch_train*(epoch-1)+batch,dataformats='NHWC')
    writer_train.add_scalar('loss',np.mean(loss_arr),epoch)

def evaluate(model,data_loader,device,num_classes):
    model.eval()
    loss_arr=[]
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            # Forward
            output = model(input)
            # Result
        print(f"TEST: LOSS {np.mean(loss_arr):.4f} | mIOU {np.mean(acc_arr):.2f}%")
if __name__=="__main__":
    from torch.utils.tensorboard import SummaryWriter
    args = get_args()
    # PATH
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir,"dataset",args.data_path)
    train_dir = os.path.join(data_dir,"train")
    val_dir = os.path.join(data_dir,"val")
    ckpt_dir = os.path.join(root_dir,"checkpoint")
    log_dir = os.path.join(root_dir,"logs")
    batch_size = args.batch_size
    num_epoch = args.epochs
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    num_workers = args.num_workers
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DataLoader
    train_ds, num_classes = get_dataset(args.data_path,args.dataset,"train",transforms=transform_train())
    test_ds, _ = get_dataset(args.data_path,args.dataset,"val",transforms=transform_eval())
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    # 모델 생성
    if args.model == "unet":
        model = Unet(num_classes=num_classes)
    # 손실 함수 정의
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 옵티마이저 정의
    optim = torch.optim.SGD(model.parameters(),lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        [int(0.5*num_epoch),int(0.75*num_epoch)],
        gamma=0.2)
    if args.test_only:
        evaluate(model,test_loader,device,num_classes)
        exit()
    
    # 부수적인 Variable
    num_data_train = len(train_ds)
    num_batch_train = int(np.ceil(num_data_train/batch_size))
    
    # Tensorboard
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir,"train"))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir,"val"))
    # 학습하던 모델 있으면 로드
    if args.resume:
        model, optim, start_epoch = load(ckpt_dir=ckpt_dir,name=args.resume,net=model,optim=optim)

    for epoch in range(start_epoch+1,num_epoch+1):
        # Train mode
        model.train()
        loss_arr = []

        for batch, data in enumerate(train_loader,start=1):
            X,Y = data[0].to(device),data[1].to(device)
            # Forwardprop
            pred = model(X)
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

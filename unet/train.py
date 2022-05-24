import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader

from models.model import Unet
from utils.Util import make_figure,make_iou_bar,save,load
from utils.Dataset import get_dataset
from utils.Transform import get_transform
from utils.Metric import IOU

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--dataset", default="voc", type=str, help="dataset name")
    parser.add_argument("--model", default="unet", type=str, help="model name")
    parser.add_argument("-j", "--num_workers", default=4, type=int, help="number of data loading workers (default: 16)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu")
    parser.add_argument("--epochs", default=30, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight-decay",default=1e-4,type=float,help="weight_decay")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--test-only",help="Only test the model",action="store_true")
    return parser.parse_args()

def train_one_epoch(model,criterion,optimizer,data_loader,lr_scheduler,epoch,best_miou):
    model.train()
    loss_arr = []
    iou_arr = []
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
        # Metric
        loss_arr.append(loss.item())
        iou = IOU(output,target,num_classes).tolist()
        iou_arr.append(iou)
        loss_mean = np.mean(loss_arr)
        miou = np.mean(iou_arr)
        # Result
        line = f"TRAIN: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_train:04d} | LOSS {loss_mean:.4f} | mIOU {miou:.4f}"
        print(line)
        logfile.write(line)
        # Tensorboard
        writer_train.add_scalar('loss',loss_mean,(epoch-1)*num_batch_train+batch)
        writer_train.add_scalar('mIOU',miou,(epoch-1)*num_batch_train+batch)
    if best_miou < miou:
        save(ckpt_dir,model,optim,epoch,miou,"model_best.pth")
        best_miou = miou
    if epoch % 30:
        save(ckpt_dir,model,optim,epoch,best_miou)
    # Tensorboard
    fig = make_figure(input,output,target,colormap)
    iou_bar = make_iou_bar(np.mean(iou_arr,axis=0),classes[1:])
    writer_train.add_figure('Images',fig,epoch)
    writer_train.add_figure('IOU',iou_bar,epoch)

def evaluate(model,criterion,data_loader,epoch=1,mode="val"):
    if mode == "val":
        header = "VALID"
    else:
        header = "TEST"
    model.eval()
    loss_arr=[]
    iou_arr=[]
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            # Forward
            output = model(input)
            # Metric
            loss = criterion(output,target)
            loss_arr.append(loss.item())
            iou = IOU(output,target,num_classes).tolist()
            iou_arr.append(iou)
            loss_mean = np.mean(loss_arr)
            miou = np.mean(iou_arr)
        # Result
        line = f"{header}: LOSS {loss_mean:.4f} | mIOU {miou:.2f}%"
        print(line)
        logfile.write(line)
        if mode == "val":
            writer_val.add_scalar('loss',loss_mean,(epoch-1)*num_batch_train)
            writer_val.add_scalar('mIOU',miou,(epoch-1)*num_batch_train)
            fig = make_figure(input,output,target,colormap)
            iou_bar = make_iou_bar(np.mean(iou_arr,axis=0),classes[1:])
            writer_val.add_figure('Images',fig,epoch)
            writer_val.add_figure('IOU',iou_bar,epoch)

if __name__=="__main__":
    import time
    import datetime
    from torch.utils.tensorboard import SummaryWriter
    args = get_args()
    # PATH
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dataset == "voc":
        data_dir = os.path.join(root_dir,"dataset","VOCdevkit","VOC2012")
    else:
        print("아직")
        exit()
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
    train_ds, num_classes = get_dataset(data_dir,"train",transforms=get_transform(train=True))
    test_ds, _ = get_dataset(data_dir,"val",transforms=get_transform(train=False))
    colormap = train_ds.colormap
    classes = train_ds.classes

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
    # 부수적인 Variable
    num_data_train = len(train_ds)
    num_batch_train = int(np.ceil(num_data_train/batch_size))
    # Tensorboard
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir,"train"))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir,"val"))
    # Train log
    logfile = open(os.path.join(log_dir,"trainlog"),"a")
    # 모델 생성
    if args.model == "unet":
        model = Unet(num_classes=num_classes)
    params = sum(p.numel for p in model.parameters if p.requires_grad)
    # 손실 함수 정의
    loss_fn = torch.nn.CrossEntropyLoss()
    # 옵티마이저 정의
    optim = torch.optim.SGD(model.parameters(),lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[int(0.5*num_epoch),int(0.75*num_epoch)],gamma=0.2)
    if args.test_only:
        model, optim, start_epoch, best_miou = load(ckpt_dir=ckpt_dir,name="model_best.pth",net=model,optim=optim)
        evaluate(model,loss_fn,test_loader,start_epoch,mode="Test")
        exit()

    # 학습하던 모델 있으면 로드
    if args.resume:
        model, optim, start_epoch, best_miou = load(ckpt_dir=ckpt_dir,name=args.resume,net=model,optim=optim)
    start_time = time.time()
    if start_epoch == 0:
        logfile.write("\nTrain Start\n")
    for epoch in range(start_epoch+1,num_epoch+1):
        train_one_epoch(model,loss_fn,optim,train_loader,lr_scheduler,epoch,best_miou)
        evaluate(model,loss_fn,test_loader,epoch,"val")
    total_time = time.time() - start_time
    writer_train.add_text("total time",str(datetime.timedelta(total_time)))
    writer_train.add_text("Parameters",str(params))
    logfile.close()
    writer_train.close()
    writer_val.close()

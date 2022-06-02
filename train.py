import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50

from models.model import Unet
from utils.Util import make_figure,make_iou_bar,save,load
from utils.Dataset import get_dataset
from utils.Transform import get_transform
from utils.Metric import intersection_union

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--dataset", default="voc", type=str, help="dataset name")
    parser.add_argument("--model", default="unet", type=str, help="model name")
    parser.add_argument("-j", "--num_workers", default=0, type=int, help="number of data loading workers (default: 0)")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="images per gpu")
    parser.add_argument("--image-size", default=512, type=int, help="input image size")
    parser.add_argument("--epochs", default=150, type=int, help="number of total epochs to run")
    parser.add_argument("--optim", default='sgd',choices=['sgd','adam'], type=str, help="optimizer")
    parser.add_argument("--lr", default=1e-2, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.2, type=float, help="momentum")
    parser.add_argument("--weight-decay",default=1e-4,type=float,help="weight_decay")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--test-only",help="Only test the model",action="store_true")
    parser.add_argument("--test-model",help="test model select",required=False)
    return parser.parse_args()

def train_one_epoch(model,criterion,optimizer,data_loaders,lr_scheduler,epoch,best_miou):
    model.train()
    for mode in ['train','val']:
        data_loader = data_loaders[mode]
        header = mode.upper()
        loss_arr = []
        total_intersection = np.zeros((num_classes,))
        total_union = np.zeros((num_classes,))
        if mode == 'train':
            writer = writer_train
            model.train()

            for batch, data in enumerate(data_loader,1):
                inputs, targets = data['input'].to(device), data['target'].to(device)
                outputs = model(inputs)['out']

                loss = criterion(outputs,targets.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Metric
                loss_arr.append(loss.item())
                loss_mean = np.mean(loss_arr)
                intersection, union = intersection_union(outputs,targets,num_classes)
                total_intersection += intersection
                total_union += union
                miou = np.nanmean(total_intersection[1:]) / np.nanmean(total_union[1:])
                # Result
                line = f"{header}: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_train:04d} | LOSS {loss_mean:.4f} | mIOU {miou:.4f}"
                print(line)
                logfile.write(line+"\n")
                # Tensorboard
                writer.add_scalar('loss',loss_mean,(epoch-1)*num_batch_train+batch)
                writer.add_scalar('mIOU',miou,(epoch-1)*num_batch_train+batch)
        else:
            writer = writer_val
            loss_mean,miou,fig,iou_bar = evaluate(model,criterion,data_loader,mode=mode)
            # Result
            line = f"{header}: EPOCH {epoch:04d} / {num_epoch:04d} | LOSS {loss_mean:.4f} | mIOU {miou:.4f}"
            print(line)
            logfile.write(line+"\n")
            # Tensorboard
            writer.add_scalar('loss',loss_mean,(epoch-1)*num_batch_train+batch)
            writer.add_scalar('mIOU',miou,(epoch-1)*num_batch_train+batch)
            if lr_scheduler is not None:
                lr_scheduler.step(loss_mean)
        if mode == 'train':
            fig = make_figure(inputs,targets,outputs,colormap)
            iou = total_intersection / total_union
            iou_bar = make_iou_bar(np.nan_to_num(iou[1:]),classes[1:])
        writer.add_figure('Images',fig,epoch)
        writer.add_figure('IOU',iou_bar,epoch)
    if mode == 'val' and best_miou < miou: # Best Model save
        save(ckpt_dir,model,optim,epoch,miou,"model_best.pth")
        best_miou = miou # Checkpoint save
    if epoch % 30 == 0:
        save(ckpt_dir,model,optim,epoch,best_miou)

def evaluate(model,criterion,data_loader,mode):
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
    iou = total_intersection / total_union
    miou = np.nanmean(total_intersection[1:]) / np.nanmean(total_union[1:])
    iou_bar = make_iou_bar(np.nan_to_num(iou[1:]),classes[1:]) # without background
    # make figure
    data = next(iter(data_loader))
    inputs, targets = data['input'].to(device), data['target'].to(device)
    outputs = model(inputs)['out']
    fig = make_figure(inputs.detach().cpu(),targets.detach().cpu(),outputs.detach().cpu(),colormap)
    return loss_mean,miou,fig,iou_bar

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
    log_dir = os.path.join(root_dir,"logs")
    os.makedirs(log_dir,exist_ok=True)
    log_count = len(os.listdir(log_dir))+1
    if args.resume:
        log_dir = os.path.join(log_dir,args.resume)
        ckpt_dir = os.path.join(root_dir,"checkpoint",args.resume)
    else:
        log_dir = os.path.join(log_dir,f'model_{log_count}')
        ckpt_dir = os.path.join(root_dir,"checkpoint",f'model_{log_count}')
    if args.test_only:
        ckpt_dir = os.path.join(root_dir,"checkpoint",args.test_model)
    batch_size = args.batch_size
    num_epoch = args.epochs
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    num_workers = args.num_workers
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DataLoader
    train_ds, num_classes = get_dataset(data_dir,args.dataset,"train",transform=get_transform(train=True,base_size=args.image_size,crop_size=int(args.image_size*0.875)))
    val_ds, _ = get_dataset(data_dir,args.dataset,"val",transform=get_transform(train=False,base_size=args.image_size))
    colormap = train_ds.colormap
    classes = train_ds.classes

    data_loaders = {'train':DataLoader(train_ds,batch_size=batch_size,num_workers=num_workers,shuffle=True),
                    'val':DataLoader(val_ds,batch_size=batch_size,num_workers=num_workers,shuffle=False)}
    # 부수적인 Variable
    num_data_train = len(train_ds)
    num_batch_train = int(np.ceil(num_data_train/batch_size))
    # 모델 생성
    if args.model == "unet":
        model = Unet(num_classes=num_classes).to(device)
    else:
        model = fcn_resnet50(pretrained=False).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 손실 함수 정의
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    # 옵티마이저 정의
    if args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(),lr=lr)
        lr_scheduler = None
    elif args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(),lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=0.2,patience=10)
    # Check Test-only
    if args.test_only:
        start=time.time()
        writer_test = SummaryWriter(log_dir=os.path.join(log_dir,"test"))
        model, optim, start_epoch, best_miou = load(ckpt_dir=ckpt_dir,model=model,optim=optim,name="model_best.pth")
        loss_mean,miou,fig,iou_bar= evaluate(model,loss_fn,data_loaders['val'],mode="test")
        print(f"TEST: LOSS {loss_mean:.4f} | mIOU {miou:.4f}")
        end = time.time()
        print(f"Average Inference Time : {(end-start)/len(val_ds)}")
        writer_test.add_figure('Images',fig)
        writer_test.add_figure('IOU',iou_bar)
        writer_test.close()
        exit()

    # 학습하던 모델 있으면 로드
    if args.resume:
        model, optim, start_epoch, best_miou = load(ckpt_dir=ckpt_dir,name=args.resume,net=model,optim=optim)
    else:
        start_epoch, best_miou = 0, 0

    # Tensorboard
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir,"train"))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir,"val"))
    # Train log
    logfile = open(os.path.join(log_dir,"trainlog.txt"),"a")
    logfile.write("Train Start : "+str(datetime.datetime.now())+"\n")
    start_time = time.time()
    for epoch in range(start_epoch+1,num_epoch+1):
        train_one_epoch(model,loss_fn,optim,data_loaders,lr_scheduler,epoch,best_miou)
    evaluate(model,loss_fn,data_loaders['val'],'test')
    total_time = time.time() - start_time
    writer_train.add_text("total time",str(datetime.timedelta(seconds=total_time)))
    writer_train.add_text("Parameters",str(params))
    logfile.write("Total Time : "+str(datetime.timedelta(seconds=total_time))+"\n")
    logfile.close()
    writer_train.close()
    writer_val.close()

from tqdm import tqdm

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50

import .models
from utils import save,load
from utils import intersection_union,make_figure,make_iou_bar
from utils import get_dataset,get_transform

def get_args():

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument("--data_root", type=str,'./dataset' help="path to Dataset",required=True)
    parser.add_argument("--dataset", choices=['voc2012','cityscapes'], type=str, help="dataset name",required=True)
    parser.add_argument("--num_classes", type=int, default=None,help="num classes (default: None)")
    available_models = sorted(name for name in models.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", choices=['fcn_resnet50','fcn_resnet101','deeplabv3plus_resnet50','deeplabv3plus_resnet101'], type=str, help="model name",required=True)
    parser.add_argument("-j", "--num_workers", default=0, type=int, help="number of data loading workers (default: 0)")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="images per gpu")
    parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--optim", default='sgd',choices=['sgd','adam'], type=str, help="optimizer")
    parser.add_argument("--lr", default=1e-2, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.5, type=float, help="momentum")
    parser.add_argument("--weight-decay",default=1e-4,type=float,help="weight_decay")
    parser.add_argument("--image-size", default=512, type=int, help="input image size")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--test-only",help="Only test the model",action="store_true")

    return parser.parse_args()

def get_optimizer(kargs):
    optims = {
        'sgd':torch.optim.SGD(model.parameters(),lr=kargs['lr'],
                            momentum=kargs['momentum'],
                            weight_decay=kargs['weight_decay']),
        'adam':torch.optim.Adam(model.parameters(),lr=kargs['lr'])
    }
    return optims[kargs['optim']]

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
                images, targets = data['image'].to(device), data['target'].to(device)
                outputs = model(images)

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
                line = f"{header}: EPOCH {epoch:04d} / {epochs:04d} | BATCH {batch:04d} / {num_batch_train:04d} | LOSS {loss_mean:.4f} | mIOU {miou:.4f}"
                print(line)
                logfile.write(line+"\n")
                # Tensorboard
                writer.add_scalar('loss',loss_mean,(epoch-1)*num_batch_train+batch)
                writer.add_scalar('mIOU',miou,(epoch-1)*num_batch_train+batch)
        else:
            writer = writer_val
            loss_mean,miou,fig,iou_bar = evaluate(model,criterion,data_loader,mode=mode)
            # Result
            line = f"{header}: EPOCH {epoch:04d} / {epochs:04d} | LOSS {loss_mean:.4f} | mIOU {miou:.4f}"
            print(line)
            logfile.write(line+"\n")
            # Tensorboard
            writer.add_scalar('loss',loss_mean,(epoch-1)*num_batch_train+batch)
            writer.add_scalar('mIOU',miou,(epoch-1)*num_batch_train+batch)
            if lr_scheduler is not None:
                lr_scheduler.step(loss_mean)
        if mode == 'train':
            fig = make_figure(images,targets,outputs,colormap)
            iou = total_intersection / total_union
            iou_bar = make_iou_bar(np.nan_to_num(iou[1:]),classes[1:])
        writer.add_figure('Images',fig,epoch)
        writer.add_figure('IOU',iou_bar,epoch)
    if mode == 'val' and best_miou < miou: # Best Model save
        save(ckpt_dir,model,optim,epoch,miou,"model_best.pth")
        best_miou = miou # Checkpoint save
    if epoch % 20 == 0:
        save(ckpt_dir,model,optim,epoch,best_miou)

if __name__=="__main__":
    import time
    import datetime
    from torch.utils.tensorboard import SummaryWriter

    kargs = vars(get_args())
    # PATH
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir,"dataset")
    log_dir = os.path.join(root_dir,"logs")
    os.makedirs(log_dir,exist_ok=True)
    if kargs['resume']:
        log_dir = os.path.join(log_dir,kargs['resume'])
        ckpt_dir = os.path.join(root_dir,"checkpoint",kargs['resume'])
    else:
        log_dir = os.path.join(log_dir,f'{kargs['model']}_{kargs['dataset']}_{kargs['optim']}')
        ckpt_dir = os.path.join(root_dir,"checkpoint",f'{kargs['model']}_{kargs['dataset']}_{kargs['optim']}')

    batch_size = kargs['batch_size']
    num_workers = kargs['num_workers']
    epochs = kargs['epochs']
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DataLoader
    train_ds = get_dataset(data_dir,kargs['dataset'],"train",transform=get_transform(kargs['dataset'],train=True))
    val_ds = get_dataset(data_dir,kargs['dataset'],"val",transform=get_transform(kargs['dataset'],train=False))
    data_loaders = {'train':DataLoader(train_ds,batch_size=batch_size,num_workers=num_workers,shuffle=True),
                    'val':DataLoader(val_ds,batch_size=batch_size,num_workers=num_workers,shuffle=False)}
    # Parameters
    colormap = train_ds.getclasses()
    classes = train_ds.getcmap()
    num_classes = len(classes)

    num_data_train = len(train_ds)
    num_batch_train = int(np.ceil(num_data_train/batch_size))
    # Model
    model = get_model(kargs['model'],num_classes).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    # Optimizer
    optim = get_optimizer(kargs)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=0.2,patience=10)

    # Load Checkpoint
    if kargs['resume']:
        model, optim, start_epoch, best_miou, time_offset = load(ckpt_dir=ckpt_dir,name=kargs['resume'],net=model,optim=optim)
    else:
        start_epoch, best_miou, time_offset = 0, 0, 0

    # Test-only
    if args.test_only:
        model, optim, _, _, _ = load(ckpt_dir=ckpt_dir,model=model,optim=optim,name="model_best.pth")
        start=time.time()
        loss_mean,miou,fig,iou_bar= evaluate(model,loss_fn,data_loaders['val'],mode="test")
        end = time.time()
        print(f"TEST: LOSS {loss_mean:.4f} | mIOU {miou:.4f}")
        print(f"Average Inference Time : {(end-start)/len(val_ds)}")
        fig.savefig('Prediction.png')
        iou_bar.savefig('IOU.png')
    else:
        # Tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(log_dir,"train"))
        writer_val = SummaryWriter(log_dir=os.path.join(log_dir,"val"))
        # Train log
        logfile = open(os.path.join(log_dir,"trainlog.txt"),"a")
        logfile.write("Train Start : "+str(datetime.datetime.now())+"\n")
        start_time = time.time()
        for epoch in range(start_epoch+1,epochs+1):
            train_one_epoch(model,loss_fn,optim,data_loaders,lr_scheduler,epoch,best_miou)
        evaluate(model,loss_fn,data_loaders['val'],'test')
        total_time = time.time() - start_time + time_offset
        writer_train.add_text("total time",str(datetime.timedelta(seconds=total_time)))
        writer_train.add_text("Parameters",str(params))
        logfile.write("Total Time : "+str(datetime.timedelta(seconds=total_time))+"\n")
        logfile.close()
        writer_train.close()
        writer_val.close()


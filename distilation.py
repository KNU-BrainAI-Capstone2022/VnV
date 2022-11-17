import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from utils.Util import load_for_distilation,label_to_one_hot_label

import models
from utils import get_dataset,mask_colorize,make_figure,make_iou_bar,save,load,SegMetrics,Denormalize

def loss_distillation(logits, labels, teacher_logits, T = 10, alpha = 0.1):
    # # Method 1
    # student_loss = F.cross_entropy(input=logits, target=labels, ignore_index=255)
    # distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
    # total_loss =  alpha*student_loss + (1-alpha)*distillation_loss
    # Method 2
    total_loss = nn.KLDivLoss()(F.log_softmax(logits/T, dim=1),
                             F.softmax(teacher_logits/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(logits, labels, ignore_index=255) * (1. - alpha)
    # # Method 3
    # student_loss = F.cross_entropy(input=logits, target=labels, ignore_index=255)
    # distillation_loss = (F.softmax(teacher_logits/T, dim=1) * (F.log_softmax(teacher_logits/T, dim=1) - F.log_softmax(logits/T, dim=1))).mean()
    # total_loss =  alpha*student_loss + (1-alpha)*distillation_loss
    
    return total_loss

def get_args():

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./dataset',help="path to Dataset")
    parser.add_argument("--dataset", choices=['voc2012','cityscapes'], type=str, help="dataset name",required=True)
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")
    parser.add_argument("--cmap", default=None, help="mask colormap (default: None)")
    parser.add_argument("-j", "--num_workers", default=0, type=int, help="number of data loading workers (default: 0)")
    parser.add_argument("--batch_size", default=8, type=int, help="images per gpu")
    parser.add_argument("--val_batch_size", default=8, type=int, help="validation images per gpu")
    parser.add_argument("--crop_size", default=512, type=int, help="input image crop size")

    # Model Options
    available_models = sorted(name for name in models.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              models.model.__dict__[name])
                              )
    parser.add_argument("--teacher", choices=available_models, default="deeplabv3plus_resnet50", type=str, help="model name",required=True)
    parser.add_argument("--student", choices=available_models, default="deeplabv3plus_mobilenet", type=str, help="model name",required=True)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16]) # DeepLab Only

    # Train Options
    parser.add_argument("--test",action="store_true",help="Only test the model")
    parser.add_argument("--save_results", action='store_true', default=False,help="save segmentation results")
    parser.add_argument("--total_iters", default=30000, type=int, help="number of total iterations to run (default: 30k)")
    parser.add_argument("--lr", default=1e-2, type=float, help="initial learning rate")
    parser.add_argument("--lr_scheduler", type=str, default=None, choices=[None, 'exp', 'step'],help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000,help="(default: 10k)")
    parser.add_argument("--weight_decay",default=1e-4,type=float,help="weight_decay")
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--print_interval", type=int, default=100,help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1000,help="iteration interval for eval (default: 100)")
    
    # Distillation Options
    parser.add_argument("--alpha", default=0.1, type=float, help="distillation loss ratio")
    return parser.parse_args()

def validate(model,criterion,dataloader,metrics,device,kargs):
    metrics.reset()
    with torch.no_grad():
        for data in dataloader:
            images = data['image'].to(device, dtype=torch.float32)
            targets = data['target'].squeeze(1).to(device, dtype=torch.long)
            outputs = model(images)
            
            loss = criterion(outputs,targets).detach().cpu().numpy()
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = targets.cpu().numpy()

            metrics.update(targets, preds, loss)
        score = metrics.get_results()
    return score

def main():
    import time
    import datetime
    from torch.utils.tensorboard import SummaryWriter

    kargs = vars(get_args())
    if kargs['dataset'].lower() == 'voc2012':
        kargs['num_classes'] = 21
    elif kargs['dataset'].lower() == 'cityscapes':
        kargs['num_classes'] = 19
    
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PATH
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir,"dataset")
    log_dir = os.path.join(root_dir,"logs_distilation")
    os.makedirs(log_dir,exist_ok=True)
    log_dir = os.path.join(log_dir,f"{kargs['student']}_new_distill_{kargs['alpha']}_{kargs['lr']}_fix_{kargs['dataset']}")
    ckpt_dir = os.path.join(root_dir,"checkpoint")
    os.makedirs(ckpt_dir,exist_ok=True)
    teacher_ckpt_dir = os.path.join(ckpt_dir,f"{kargs['teacher']}_{kargs['dataset']}")
    ckpt_dir = os.path.join(ckpt_dir,f"{kargs['student']}_new_distill_{kargs['alpha']}_{kargs['lr']}_fix_{kargs['dataset']}")

    # DataLoader
    train_ds, val_ds= get_dataset(data_dir,kargs)
    dataloaders = {'train':DataLoader(train_ds,batch_size=kargs['batch_size'],num_workers=kargs['num_workers'],shuffle=True),
                    'val':DataLoader(val_ds,batch_size=kargs['val_batch_size'],num_workers=kargs['num_workers'],shuffle=False)}
    kargs['cmap'] = train_ds.getcmap()
    
    # Student, Teacher Model
    student = models.model.__dict__[kargs['student']](num_classes=kargs['num_classes'],output_stride=kargs['output_stride'],pretrained_backbone=False).to(device)
    teacher = models.model.__dict__[kargs['teacher']](num_classes=kargs['num_classes'],output_stride=kargs['output_stride']).to(device)
        
    # Optimizer
    optimizer = torch.optim.SGD(params=student.parameters(), lr=kargs['lr'], momentum=0.9, weight_decay=kargs['weight_decay'])
    
    if kargs['lr_scheduler'] == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif kargs['lr_scheduler'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=kargs['step_size'], gamma=0.1)
    else:
        lr_scheduler = None

    # Load Checkpoint
    student, teacher, optimizer, lr_scheduler, cur_iter, best_score, = load_for_distilation(ckpt_dir,teacher_ckpt_dir,student,teacher,optimizer,lr_scheduler,kargs)
    # Metric
    metrics = SegMetrics(kargs['num_classes'])
    # Loss (Validate)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    # Test-only
    if kargs['test']:
        start=time.time()
        student.eval()
        val_score = validate(student,criterion,dataloaders['val'],metrics,device,kargs)
        end = time.time()
        print(metrics.to_str(val_score))
        print(f"Average Inference Time : {(end-start)/len(val_ds)}")
    else:
        # Tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(log_dir,"train"))
        writer_val = SummaryWriter(log_dir=os.path.join(log_dir,"val"))

        # Train
        start_time = time.time()
        epoch = 0
        while True:
            student.train()
            teacher.eval()
            epoch += 1
            interval_loss = 0.0
            for data in dataloaders['train']:
                cur_iter += 1
                if cur_iter > kargs['total_iters']:
                    break
                images = data['image'].to(device,dtype=torch.float32)
                targets = data['target'].to(device,dtype=torch.long)
                
                outputs = student(images)
                teacher_outputs = teacher(images)

                loss = loss_distillation(outputs,targets,teacher_outputs,alpha=kargs['alpha'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Metric
                interval_loss += loss.detach().cpu().numpy()
                # Tensorboard
                if cur_iter % kargs['print_interval'] == 0:
                    interval_loss = interval_loss / kargs['print_interval']
                    print(f"EPOCH {epoch} | ITERATION {cur_iter} / {kargs['total_iters']:d} | LOSS {interval_loss:.4f}")
                    writer_train.add_scalar('loss',interval_loss,cur_iter)
                    interval_loss = 0.0
                if cur_iter % kargs['val_interval'] == 0:
                    student.eval()
                    print("Validation")
                    val_score = validate(student,criterion,dataloaders['val'],metrics,device,kargs)
                    print(metrics.to_str(val_score))
                    save(ckpt_dir,student,optimizer,lr_scheduler,cur_iter,best_score,f"model_{str(cur_iter).rjust(6,'0')}.pth")
                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        save(ckpt_dir,student,optimizer,lr_scheduler,cur_iter,best_score,"model_best.pth")
                    # writer_val.add_scalar('Mean Acc',val_score['Mean Acc'],cur_iter)
                    writer_val.add_scalar('loss',val_score['Mean Loss'],cur_iter)
                    writer_val.add_scalar('mIOU',val_score['Mean IoU'],cur_iter)
                    fig = make_figure(images.detach().cpu(),targets.cpu(),outputs.detach().cpu(),kargs['cmap'])
                    iou_bar = make_iou_bar(np.nan_to_num(val_score['Class IoU'].values()))
                    writer_val.add_figure('Images',fig,cur_iter)
                    writer_val.add_figure('Class IOU',iou_bar,cur_iter)
                    student.train()
                if cur_iter > 30000 and lr_scheduler: # 30000부터 step lr 적용
                    lr_scheduler.step()
            if cur_iter > kargs['total_iters']:
                break
        # total_time = time.time() - start_time + time_offset
        # writer_train.add_text("total time",str(datetime.timedelta(seconds=total_time)))
        writer_train.close()
        writer_val.close()

if __name__=="__main__":
    main()
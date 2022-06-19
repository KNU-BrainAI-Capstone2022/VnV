from tqdm import tqdm

import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader

import models
from utils import get_dataset,mask_colorize,make_figure,make_iou_bar,save,load,SegMetrics,Denormalize

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
    parser.add_argument("--model", choices=available_models, default="deeplabv3plus_resnet50", type=str, help="model name",required=True)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16]) # DeepLab Only

    # Train Options
    parser.add_argument("--test_only",action="store_true",help="Only test the model")
    parser.add_argument("--save_results", action='store_true', default=False,help="save segmentation results")
    parser.add_argument("--total_iters", default=30000, type=int, help="number of total iterations to run (default: 30k)")
    parser.add_argument("--lr", default=1e-2, type=float, help="initial learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='step', choices=['exp', 'step'],help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000,help="(default: 10k)")
    parser.add_argument("--weight_decay",default=1e-4,type=float,help="weight_decay")
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--print_interval", type=int, default=100,help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1000,help="iteration interval for eval (default: 100)")

    return parser.parse_args()

def validate(model,criterion,dataloader,metrics,device,kargs):
    metrics.reset()
    if kargs['save_results']:
        if not os.path.exists('results',f"{kargs['model']}_{kargs['dataset']}"):
            os.mkdir('results',f"{kargs['model']}_{kargs['dataset']}")
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_id = 0
    with torch.no_grad():
        for data in dataloader:
            images = data['image'].to(device, dtype=torch.float32)
            targets = data['target'].squeeze(1).to(device, dtype=torch.long)
            outputs = model(images)

            loss = criterion(outputs,targets).detach().cpu().numpy()
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = targets.cpu().numpy()

            metrics.update(targets, preds, loss)
            break
            if kargs['save_results']:
                for i in range(1):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = mask_colorize(target,kargs['cmap']).astype(np.uint8)
                    pred = mask_colorize(pred,kargs['cmap']).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
                    ax.yaxis.set_major_locator(mpl.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

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
    log_dir = os.path.join(root_dir,"logs")
    os.makedirs(log_dir,exist_ok=True)
    log_dir = os.path.join(log_dir,f"{kargs['model']}_{kargs['dataset']}")
    ckpt_dir = os.path.join(root_dir,"checkpoint")
    os.makedirs(ckpt_dir,exist_ok=True)
    ckpt_dir = os.path.join(ckpt_dir,f"{kargs['model']}_{kargs['dataset']}")

    # DataLoader
    train_ds, val_ds= get_dataset(data_dir,kargs)
    dataloaders = {'train':DataLoader(train_ds,batch_size=kargs['batch_size'],num_workers=kargs['num_workers'],shuffle=True),
                    'val':DataLoader(val_ds,batch_size=kargs['val_batch_size'],num_workers=kargs['num_workers'],shuffle=False)}
    kargs['cmap'] = train_ds.getcmap()
    # Model
    model = models.model.__dict__[kargs['model']](num_classes=kargs['num_classes'],output_stride=kargs['output_stride']).to(device)
    # Loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    # Optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * kargs['lr']},
        {'params': model.classifier.parameters(), 'lr': kargs['lr']},
    ], lr=kargs['lr'], momentum=0.9, weight_decay=kargs['weight_decay'])
    if kargs['lr_scheduler'] == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif kargs['lr_scheduler'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=kargs['step_size'], gamma=0.1)

    # Load Checkpoint
    model, optimizer, lr_scheduler, cur_iter, best_score, = load(ckpt_dir,model,optimizer,lr_scheduler,kargs)
    # Metric
    metrics = SegMetrics(kargs['num_classes'])
    # Test-only
    if kargs['test_only']:
        start=time.time()
        model.eval()
        val_score = validate(model,criterion,dataloaders['val'],metrics,device,kargs)
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
            model.train()
            epoch += 1
            interval_loss = 0.0
            if cur_iter > kargs['total_iters']:
                break
            for data in dataloaders['train']:
                cur_iter += 1
                images = data['image'].to(device,dtype=torch.float32)
                targets = data['target'].squeeze(1).to(device,dtype=torch.long)
                outputs = model(images)

                optimizer.zero_grad()
                loss = criterion(outputs,targets)
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
                    model.eval()
                    print("Validation")
                    val_score = validate(model,criterion,dataloaders['val'],metrics,device,kargs)
                    print(metrics.to_str(val_score))
                    save(ckpt_dir,model,optimizer,lr_scheduler,cur_iter,best_score,"model_last.pth")
                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        save(ckpt_dir,model,optimizer,lr_scheduler,cur_iter,best_score,"model_best.pth")
                    # writer_val.add_scalar('Mean Acc',val_score['Mean Acc'],cur_iter)
                    writer_val.add_scalar('loss',val_score['Mean Loss'],cur_iter)
                    writer_val.add_scalar('mIOU',val_score['Mean IoU'],cur_iter)
                    images = images.detach().cpu()
                    fig = make_figure(images.detach().cpu(),targets.cpu(),outputs.detach().cpu(),kargs['cmap'])
                    iou_bar = make_iou_bar(np.nan_to_num(val_score['Class IoU'].values()))
                    writer_val.add_figure('Images',fig,cur_iter)
                    writer_val.add_figure('Class IOU',iou_bar,cur_iter)
                    model.train()
            lr_scheduler.step()
        total_time = time.time() - start_time + time_offset
        writer_train.add_text("total time",str(datetime.timedelta(seconds=total_time)))
        writer_train.close()
        writer_val.close()

if __name__=="__main__":
    main()
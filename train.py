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
    # Dataset Options
    parser.add_argument("--data_root", type=str,'./dataset' help="path to Dataset",required=True)
    parser.add_argument("--dataset", choices=['voc2012','cityscapes'], type=str, help="dataset name",required=True)
    parser.add_argument("--num_classes", type=int, default=None,help="num classes (default: None)")

    available_models = sorted(name for name in models.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              models.model.__dict__[name])
                              )
    # Model Options
    parser.add_argument("--model", choices=available_models, type=str, help="model name",required=True)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only",action="store_true",help="Only test the model")
    parser.add_argument("--save_results", action='store_true', default=False,help="save segmentation results")
    parser.add_argument("--total_iters", default=3e5, type=int, help="number of total iterations to run (default: 30k)")
    parser.add_argument("-j", "--num_workers", default=0, type=int, help="number of data loading workers (default: 0)")
    parser.add_argument("--batch_size", default=8, type=int, help="images per gpu")
    parser.add_argument("--val_batch_size", default=8, type=int, help="images per gpu")
    parser.add_argument("--lr", default=1e-2, type=float, help="initial learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='exp', choices=['exp', 'step'],help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=1e4,help="(default: 10k)")
    parser.add_argument("--weight_decay",default=1e-4,type=float,help="weight_decay")
    parser.add_argument("--crop_size", default=512, type=int, help="input image crop size")
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--print_interval", type=int, default=10,help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,help="iteration interval for eval (default: 100)")

    return parser.parse_args()

def validate(args,model,dataloader,device,kargs):
    metrics.reset()
    model.eval()
    if kargs['save_results']:
        if not os.path.exists('results'):
            os.mkdir('results')
    img_id = 0
    with torch.no_grad():
        for batch, data in tqdm(enumerate(dataloader)):
            images = data['image'].to(device, dtype=torch.float32)
            labels = label['target'].to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if kargs['save_results']:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
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
    log_dir = os.path.join(log_dir,f'{kargs['model']}_{kargs['dataset']}')
    ckpt_dir = os.path.join(root_dir,"checkpoint",f'{kargs['model']}_{kargs['dataset']}')

    # DataLoader
    train_ds, val_ds= get_dataset(data_dir,kargs)
    dataloaders = {'train':DataLoader(train_ds,batch_size=kargs['batch_size'],num_workers=kargs['num_workers'],shuffle=True),
                    'val':DataLoader(val_ds,batch_size=kargs['batch_size'],num_workers=kargs['num_workers'],shuffle=False)}
    # Parameters
    total_iters = kargs['total_iters']
    colormap = train_ds.getcmap()

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
    model, optimizer, lr_scheduler, cur_iter, best_score, time_offset = load(ckpt_dir,model,optimizer,lr_scheduler,kargs)
    # Metric
    metrics = SegMetrics(kargs['num_classes'])
    # Test-only
    if args.test_only:
        start=time.time()
        val_score = validate(model,criterion,dataloaders['val'],metrics,device,kargs)
        end = time.time()
        print(metrics.to_str(val_score))
        print(f"Average Inference Time : {(end-start)/len(val_ds)}")
    else:
        # Tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(log_dir,"train"))
        writer_val = SummaryWriter(log_dir=os.path.join(log_dir,"val"))
        # Train log
        start_time = time.time()
        epoch = 0
        while True:
            model.train()
            epoch += 1
            interval_loss = 0.0
            for batch, data in enumerate(dataloader,1):
                cur_iter += 1
                images = data['image'].to(device,dtype=torch.float32)
                targets = data['target'].to(device,dtype=torch.long)
                outputs = model(images)

                optimizer.zero_grad()
                loss = criterion(outputs,targets)
                loss.backward()
                optimizer.step()
                # Metric
                interval_loss += loss.detach().cpu().numpy()
                # Tensorboard
                writer_train.add_scalar('loss',loss_mean,cur_iter)
                if cur_iter % kargs['print_interval'] == 0:
                    print("EPOCH {epoch} | ITERATION {cur_iter} / {kargs['total_iters']} | LOSS {interval_loss / 10:.4f}")
                if cur_iter % kargs['val_interval'] == 0:
                    val_score = validate(model,criterion,dataloaders['val'],metrics,device,kargs)
                    print(metrics.to_str(val_score))
                    save(ckpt_dir,model,optim,lr_scheduler,cur_iter,best_score,time,"model_last.pth")
                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        save(ckpt_dir,model,optim,lr_scheduler,cur_iter,best_score,time,"model_best.pth")
                    writer_val.add_scalar('Mean Acc',val_score['Mean Acc'],cur_iter)
                    writer_val.add_scalar('mIOU',val_score['Mean IoU'],cur_iter)
                    if cur_iter % 1000 == 0
                        fig = make_figure(images.detach().cpu(),targets.cpu(),outputs.detach().cpu(),colormap)
                        iou_bar = make_iou_bar(np.nan_to_num(val_score['Class IoU']))
                        writer_val.add_figure('Images',fig,cur_iter)
                        writer_val.add_figure('IOU',iou_bar,cur_iter)
                    model.train()
            lr_scheduler.step()
        total_time = time.time() - start_time + time_offset
        writer_train.add_text("total time",str(datetime.timedelta(seconds=total_time)))
        writer_train.close()
        writer_val.close()

if __name__=="__main__":
    main()
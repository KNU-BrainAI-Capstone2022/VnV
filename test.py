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
from torch2trt import TRTModule

def get_args():

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Test")
    # Main Options
    parser.add_argument("--checkpoint", default='', type=str, help="checkpoint path")
    parser.add_argument("--save_results", action='store_true', default=False,help="save segmentation results")
    
    # Sub Options (usually use default)
    parser.add_argument("--data_root", type=str, default='./dataset',help="path to Dataset")
    
    # tensorrt Options
    parser.add_argument("--trt", action="store_true", help="Can use tensorrt")
    parser.add_argument("--tr_path", type=str, help="Tensorrt weight path")
    
    # ---- argument processing ----
    args = parser.parse_args()
    
    args.test = True
    if args.checkpoint:
        args.modelname = args.checkpoint.split('/')[-2]
        ckptname = args.modelname.split('_')
        args.model =  '_'.join(ckptname[0:2])
        args.dataset = ckptname[-1]
    else:
        args.dataset = 'cityscapes'
    
    if args.dataset.lower() == 'voc2012':
        args.num_classes = 21
    elif args.dataset.lower() == 'cityscapes':
        args.num_classes = 19
    return args

def test(model,criterion,dataloader,metrics,device,kargs):
    metrics.reset()
    if kargs['save_results']:
        result_pth = os.path.join(os.getcwd(),'results')
        if not os.path.exists(result_pth):
            os.mkdir(result_pth)
        result_pth= os.path.join(result_pth,f"{kargs['modelname']}")
        if not os.path.exists(result_pth):
            os.mkdir(result_pth)
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_id = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data['image'].to(device, dtype=torch.float32)
            targets = data['target'].squeeze(1).to(device, dtype=torch.long)
            outputs = model(images)

            loss = criterion(outputs,targets).detach().cpu().numpy()
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = targets.cpu().numpy()

            metrics.update(targets, preds, loss)
            
            if kargs['save_results']:
                if i % 100 == 0:
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = mask_colorize(target,kargs['cmap']).astype(np.uint8)
                    pred = mask_colorize(pred,kargs['cmap']).astype(np.uint8)

                    Image.fromarray(image).save(os.path.join(result_pth,'%d_image.png' % img_id))
                    Image.fromarray(target).save(os.path.join(result_pth,'%d_target.png' % img_id))
                    Image.fromarray(pred).save(os.path.join(result_pth,'%d_pred.png' % img_id))

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
                    ax.yaxis.set_major_locator(mpl.ticker.NullLocator())
                    plt.savefig(os.path.join(result_pth,'%d_overlay.png' % img_id), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score

def main():
    import time
    from torch.utils.tensorboard import SummaryWriter

    kargs = vars(get_args())
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PATH
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir,kargs['data_root'])

    # DataLoader
    ds = get_dataset(data_dir,kargs)
    dataloader = DataLoader(ds,batch_size=1,num_workers=0,shuffle=False)
    kargs['cmap'] = ds.getcmap()
    
    # Model
    if kargs['trt']:
        if not os.path.exists(kargs['tr_path']):
            print('model is not exist\n')
            exit(1)
        model = TRTModule()
        model.load_state_dict(torch.load(kargs['tr_path']))
    else:
        model = models.model.__dict__[kargs['model']](num_classes=kargs['num_classes']).to(device)
        # Load Checkpoint
        dict_model = torch.load(kargs['checkpoint'])
        model.load_state_dict(dict_model['model_state'])
        cur_iter = dict_model['cur_iter']
        best_score = dict_model['best_score']
        print(cur_iter,best_score)
    # Loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    # Metric
    metrics = SegMetrics(kargs['num_classes'])
    
    # Test
    start=time.time()
    model.eval()
    test_score = test(model,criterion,dataloader,metrics,device,kargs)
    end = time.time()
    print(metrics.to_str(test_score))
    print(f"Average Inference Time : {(end-start)/len(ds)}")
if __name__ == '__main__':
    main()
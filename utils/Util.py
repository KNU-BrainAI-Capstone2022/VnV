import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# -------------------------- Model -----------------------------------
# 모델 저장 함수
def save(ckpt_dir,model,optim,lr_scheduler,cur_iter,best_score,filename):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    filepath = os.path.join(ckpt_dir,filename)
    torch.save({
        'cur_iter':cur_iter,
        'model_state':model.state_dict(),
        'optim_state':optim.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        'best_score':best_score,
        },filepath)
    print("Model saved as %s" % filepath)

# 모델 로드 함수
def load(ckpt_dir,model,optim,lr_scheduler,kargs):
    if kargs['resume'] == True:
        ckpt = os.path.join(ckpt_dir,"model_last.pth")
    elif kargs['test_only'] == True:
        ckpt = os.path.join(ckpt_dir,"model_best.pth")
    else:
        cur_iter = 0
        best_score = 0
        return model,optim,lr_scheduler,cur_iter,best_score

    if not os.path.exists(ckpt):
        cur_iter = 0
        best_score = 0
        print("There is no checkpoint")
        return model,optim,lr_scheduler,cur_iter,best_score

    dict_model = torch.load(ckpt)
    model.load_state_dict(dict_model['model_state'])
    optim.load_state_dict(dict_model['optim_state'])
    lr_scheduler.load_state_dict(dict_model['lr_scheduler_state'])
    cur_iter = dict_model['cur_iter']
    best_score = dict_model['best_score']
    print("Model restored from %s" % ckpt)
    return model,optim,lr_scheduler,cur_iter,best_score
# -------------------------- Model -----------------------------------

# -------------------------- Metric / Result -----------------------------------
def mask_colorize(masks,cmap):
    # masks : BxCxHxW
    # if C != 1, argmax
    if masks.size(1) == len(cmap):
        _, masks = masks.max(dim=1)
        masks = masks.unsqueeze(dim=1)
    cmap_ = torch.tensor(cmap)
    r_mask = torch.zeros_like(masks,dtype=torch.uint8)
    g_mask = torch.zeros_like(masks,dtype=torch.uint8)
    b_mask = torch.zeros_like(masks,dtype=torch.uint8)
    for k in range(len(cmap)):
        indices = masks == k
        r_mask[indices] = cmap_[k,0]
        g_mask[indices] = cmap_[k,1]
        b_mask[indices] = cmap_[k,2]
    return torch.cat([r_mask,g_mask,b_mask],dim=1)

def make_figure(images,targets,outputs,colormap):
    if targets.dim() == 3: # BxHxW
        targets = targets.unsqueeze(1)
    n=images.size(0)
    fig, ax = plt.subplots(3,1,figsize=(n*3,9))
    ax[0].imshow(torchvision.utils.make_grid(images.cpu(), normalize=True).permute(1,2,0))
    ax[0].set_title("Input")
    ax[0].axis('off')
    targets = mask_colorize(targets,colormap)
    ax[1].imshow(torchvision.utils.make_grid(targets.cpu(), normalize=False).permute(1,2,0))
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')
    outputs = mask_colorize(outputs,colormap)
    ax[2].imshow(torchvision.utils.make_grid(outputs.cpu(), normalize=False).permute(1,2,0))
    ax[2].set_title("Prediction")
    ax[2].axis('off')
    fig.tight_layout()
    return fig

def intersection_union(output:torch.Tensor,target:torch.Tensor,c:int,ignore_index=255):
    # output shape : BxCxHxW float
    # Target shape : BxHxW long
    assert output.dim() == 4
    assert target.dim() == 3

    output = torch.nn.functional.softmax(output,dim=1)
    _,output = torch.max(output,dim=1) # BxCxHxW -> BxHxW

    output = output.contiguous().view(-1).type(torch.float)
    target = target.contiguous().view(-1).type(torch.float)
    intersection = output[output==target]
    # Ignore Background
    area_intersection = torch.histc(intersection,bins=c,min=0,max=c-1)
    area_output = torch.histc(output,bins=c,min=0,max=c-1)
    area_target = torch.histc(target,bins=c,min=0,max=c-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.detach().cpu().numpy(),area_union.detach().cpu().numpy()

def make_iou_bar(cls_iou):
    fig = plt.figure()
    plt.bar(range(len(cls_iou)),cls_iou)
    fig.suptitle("IOU per Class")
    return fig

class SegMetrics(object):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean iou
            - class iou
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        cls_iou = dict(zip(range(self.num_classes), iou))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Mean IoU": mean_iou,
                "Class IoU": cls_iou,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
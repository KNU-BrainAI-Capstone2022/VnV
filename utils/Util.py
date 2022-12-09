import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from torchvision.transforms.functional import normalize
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
        for f in sorted(os.listdir(ckpt_dir),reverse=True):
            if f.find("last") != -1:
                ckpt = os.path.join(ckpt_dir,f)
                break
    elif kargs['test'] == True:
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

# Knowledge Distilation 학습 시 모델 로드 함수
def load_for_distilation(ckpt_dir,teacher_ckpt_dir,student,teacher,optim,lr_scheduler,kargs):
    t_ckpt = os.path.join(teacher_ckpt_dir,"model_best.pth")
    teacher.load_state_dict(torch.load(t_ckpt)['model_state'])
    
    if kargs['resume']:
        for f in sorted(os.listdir(ckpt_dir),reverse=True):
            if f.find("best") == -1:
                s_ckpt = os.path.join(ckpt_dir,f)
                break
    else:
        cur_iter = 0
        best_score = 0
        return student,teacher,optim,lr_scheduler,cur_iter,best_score

    dict_model = torch.load(s_ckpt)
    student.load_state_dict(dict_model['model_state'])
    optim.load_state_dict(dict_model['optim_state'])
    lr_scheduler.load_state_dict(dict_model['lr_scheduler_state'])
    cur_iter = dict_model['cur_iter']
    best_score = dict_model['best_score']
    print("Model restored from %s" % s_ckpt)
    return student,teacher,optim,lr_scheduler,cur_iter,best_score
# -------------------------- Model -----------------------------------

# -------------------------- Metric / Result -----------------------------------
class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def mask_colorize(masks,cmap):
    """
    Args:
        img (np.ndarray): np.ndarray of shape HxWxC and dtype uint8 (BGR Image)
        mask (np.ndarray): np.ndarray of shape: HxW and dtype int range of [0,num_classes).
        cmap (np.ndarray) : np.ndarray containing the colors of shape NUM_CLASSES x C and its dtype uint8 (order : RGB)
    Returns:
        newimg (np.ndarray[CxHxW]): Image np.ndarray, with segmentation masks drawn on top.
    """
    # masks : BxCxHxW
    # if C != 1, argmax
    # B H W -> B C H W (Torch.Tensor)
    # H W -> H W C(numpy.ndarray)
    if torch.is_tensor(masks):
        assert masks.ndim >= 3
        if masks.ndim == 4: # B C H W Tensor
            masks = masks.max(dim=1)[1] # B C H W -> B H W
        masks = masks.unsqueeze(1)
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
    elif isinstance(masks,np.ndarray): # H W
        assert masks.ndim == 2
        r_mask = np.zeros_like(masks,dtype=np.uint8)
        g_mask = np.zeros_like(masks,dtype=np.uint8)
        b_mask = np.zeros_like(masks,dtype=np.uint8)
        for k in range(len(cmap)):
            indices = masks == k
            r_mask[indices] = cmap[k,0]
            g_mask[indices] = cmap[k,1]
            b_mask[indices] = cmap[k,2]
        return np.stack([b_mask,g_mask,r_mask],axis=2)

def make_figure(images,targets,outputs,colormap):
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
    plt.xticks(range(len(cls_iou)))
    fig.suptitle("IOU per Class")
    return fig

class SegMetrics(object):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.loss = []
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, label_trues, label_preds, loss):
        self.loss.append(loss)
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
            - mean loss
            - overall accuracy
            - mean accuracy
            - mean iou
            - class iou
        """
        mean_loss = np.mean(self.loss)
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        cls_iou = dict(zip(range(self.num_classes), iou))

        return {
                "Mean Loss" : mean_loss,
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Mean IoU": mean_iou,
                "Class IoU": cls_iou,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.loss = []

def label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.long,
    eps: float = 1e-6,
    ignore_index = 255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    """
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
    
    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
    
    return ret
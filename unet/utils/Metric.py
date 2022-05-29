import torch

def label_to_one_hot_label(labels:torch.Tensor,num_classes:int,ignore_index=255):
    # BxHxW -> BxCxHxW
    shape = labels.shape
    labels[labels==ignore_index] = 0 # Ignore index 제거
    one_hot = torch.zeros((shape[0],num_classes)+shape[1:],dtype=torch.float32)
    one_hot.scatter_(1,labels.unsqueeze(1),1.0)
    return one_hot

def IOU(output:torch.Tensor,target:torch.Tensor,c:int,ignore_index=255):
    # output shape : BxCxHxW float
    # Target shape : BxHxW long
    assert output.dim() == 4
    assert target.dim() == 3
    eps = 1e-9 # avoid zero devision

    output = torch.nn.functional.softmax(output,dim=1)
    _,output = torch.max(output,dim=1) # BxCxHxW -> BxHxW

    output = output.contiguous().view(-1).type(torch.float)
    target = target.contiguous().view(-1).type(torch.float)
    intersection = output[output==target]
    # Background는 제외한다.
    area_intersection = torch.histc(intersection,bins=c,min=0,max=c-1)[1:]
    area_output = torch.histc(output,bins=c,min=0,max=c-1)[1:]
    area_target = torch.histc(target,bins=c,min=0,max=c-1)[1:]
    area_union = area_output + area_target - area_intersection
    iou = area_intersection / area_union
    return iou.tolist()
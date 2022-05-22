import os
import torch

# 모델 저장 함수
def save(ckpt_dir,net,optim,epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},
               os.path.join(ckpt_dir,f"model_epoch{epoch}.pth"))

# 모델 로드 함수
def load(ckpt_dir,net,optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net,optim,epoch
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda f:int(''.join(filter(str.isdigit,f))))

    dict_model = torch.load(os.path.join(ckpt_dir,ckpt_list[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(''.join(filter(str.isdigit,ckpt_list[-1])))
    return net,optim,epoch
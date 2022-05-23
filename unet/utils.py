import os
import torch

# 모델 저장 함수
def save(ckpt_dir,net,optim,epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},
               os.path.join(ckpt_dir,f"model_epoch{epoch}.pth"))

# 모델 로드 함수
def load(ckpt_dir,name,net,optim):
    ckpt = os.path.join(ckpt_dir,name)
    if not os.path.exists(ckpt):
        epoch = 0
        print("There is no checkpoint")
        return net,optim,epoch

    dict_model = torch.load(ckpt)

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(''.join(filter(str.isdigit,name)))
    return net,optim,epoch
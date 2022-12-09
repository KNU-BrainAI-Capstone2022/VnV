import torch.nn as nn
import torch

class WrappedModel(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
    
    def forward(self,x):
        x = x.permute(0,3,1,2) # 1x3xHxW
        x = x.flip(1)
        x = x / 255.0
        # x = self.model(x)
        x = self.model(x)['out']
        # x = torch.argmax(x, dim=1)
        x = torch.topk(x,k=1,dim=1,sorted=False)[1] # 1x1xHxW
        return x
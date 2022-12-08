import torch.nn as nn
import torchvision.transforms.functional as F
import torch
class WrappedModel(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
        # self.pre_layers = nn.Sequential(
        #     nn.
        # )
    
    def forward(self,x):
        x = x.flip(-1)
        x = x.permute(0,3,1,2)
        x = x / 255.0
        x = self.model(x)
        x = nn.functional.softmax(x,dim=1)
        x = torch.topk(x,1,dim=1,stored=False)[1]
        return x

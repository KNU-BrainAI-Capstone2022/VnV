import torch
from torch import nn

class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.argmax = torch.argmax
        
    def forward(self,x):
        x = self.model(x)
        x = self.argmax(x,dim=0)
        return x
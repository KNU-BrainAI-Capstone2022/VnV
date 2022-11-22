import torch
from torch import nn

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel,self).__init__()
        self.model = model
        self.argmax = torch.argmax
        
    def forward(self,x):
        x = self.model(x)
        x = self.argmax(x,dim=1)
        return x

class TestModel(nn.Module):
    def __init__(self,num_classes,output_shape):
        super(TestModel,self).__init__()
        self.num_classes = num_classes
        self.h,self.w = output_shape[2:]
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.num_classes, kernel_size=3, padding=1),
        )
        self.argmax = torch.argmax
        
    def custom_argmax(self,x):
        return torch.argmax(x,dim=1).type(torch.half)
        
    def forward(self,x):
        x = self.layer(x)
        x = self.custom_argmax(x)            
        return x
    
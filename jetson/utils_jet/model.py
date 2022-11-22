import torch
import sys
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
        y = torch.zeros((self.h,self.w),dtype=torch.half)
        for i in range(self.h):
            for j in range(self.w):
                max_idx = 0
                max_val = -float("inf")
                for k in range(self.num_classes):
                    if max_val > x[0][k][i][j]:
                        max_idx, max_val = k, x[0][k][i][j]
                y[i][j] = max_idx
        return y
        
    def forward(self,x):
        x = self.layer(x)
        x = self.custom_argmax(x)            
        return x
    
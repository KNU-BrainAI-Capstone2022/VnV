import os
import torch
from torch import nn
import utils

class FCN8(nn.Module):
    # vgg 16 
    def __init__(self,num_class=21):
        super(FCN8, self).__init__()

        self.classifier3 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4096,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096,4096,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096,num_class,kernel_size=1),
        )

        self.pool4_conv = nn.Conv2d(512,21,kernel_size=1)
        self.pool3_conv = nn.Conv2d(256,21,kernel_size=1)

        self.upsample2 = nn.ConvTranspose2d(num_class,num_class,2,2,bias=False)
        self.pool4_upsample2 = nn.ConvTranspose2d(num_class,num_class,2,2,bias=False)
        self.upsample8 = nn.ConvTranspose2d(num_class,num_class,8,8,bias=False)

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self._initialize_weights()

    def forward(self, features):
    
        pool3 = features['layer2']
        pool4 = features['layer3']
        pool5 = features['layer4']
        
        pool5 = self.classifier3(pool5)

        # 1/32 *2 + 1/16
        pool5 = self.upsample2(pool5)
        pool4 = self.pool4_conv(pool4)
        x = pool5 + pool4
        x = self.relu(x)
        
        # 1/16 *2 + 1/8
        x = self.pool4_upsample2(x)
        pool3 = self.pool3_conv(pool3)
        x = x + pool3
        x = self.relu(x)

        # 1/8 * 8 
        x = self.upsample8(x)
        return x 

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.zero_()
                    if m.bias is not None:
                        m.bias.data.zero_()

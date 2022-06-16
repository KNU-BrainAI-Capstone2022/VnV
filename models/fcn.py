import os
import torch
from torch import nn
import utils

class FCN8(nn.Module):
    # vgg 16 
    def __init__(self,backbone,num_class=21):
        super(FCN8, self).__init__()
        self.backbone = backbone

        self.fc1 = nn.Conv2d(in_channels=512,out_channels=4096,kernel_size=1)
        self.fc2 = nn.Conv2d(4096,4096,kernel_size=1)
        self.fc3 = nn.Conv2d(4096,num_class,kernel_size=1)
        self.pool4_conv = nn.Conv2d(512,21,kernel_size=1)
        self.pool3_conv = nn.Conv2d(256,21,kernel_size=1)

        self.upsample2 = nn.ConvTranspose2d(num_class,num_class,2,2,bias=False)
        self.pool4_upsample2 = nn.ConvTranspose2d(num_class,num_class,2,2,bias=False)
        self.upsample8 = nn.ConvTranspose2d(num_class,num_class,8,8,bias=False)

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        back_layers = utils.IntermediateLayerGetter(self.backbone, {'layer2': 1, 'layer3': 2, 'layer4': 3})
        pool = back_layers(x)
        pool = list(pool.values())

        pool3 = pool[0]
        pool4 = pool[1]
        pool5 = pool[2]
        
        pool5 = self.fc1(pool5)
        pool5 = self.relu(pool5)
        pool5 = self.dropout(pool5)
        pool5 = self.fc2(pool5)
        pool5 = self.relu(pool5)
        pool5 = self.dropout(pool5)
        pool5 = self.fc3(pool5)

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

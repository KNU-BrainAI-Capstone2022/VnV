import os
import torch
from torch import nn
from torch.nn import functional as F

class FCN(nn.Module):
    def __init__(self,backbone,classifier,aux_classifier=None):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        x = features['layer4']
        x = self.classifier(x)
        result = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        if self.aux_classifier is not None:
            x = features['layer3']
            x = self.aux_classifier(x)
            b = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result = 0.5*result +0.5*b 
        
        return result
    
class FCNHead(nn.Sequential):
    def __init__(self,in_channels, num_class=21):
        inter_channels = in_channels //4
        layers= [
            nn.Conv2d(in_channels,inter_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels,num_class,1),
        ]
        super().__init__(*layers)

class FCN8(nn.Module):
    def __init__(self,in_channels,num_class=21):
        super(FCN8, self).__init__()
        pro_channell = in_channels //4
        pro_channel2 = in_channels //2
        
        self.project1 = nn.Sequential(
            nn.Conv2d(pro_channell,256,kernel_size=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256,num_class,kernel_size=1),
        )

        self.project2 = nn.Sequential(
            nn.Conv2d(pro_channel2,512,kernel_size=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512,num_class,kernel_size=1),
        )

        self.classifier3 = nn.Sequential(
            nn.Conv2d(in_channels,512,kernel_size=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=512,out_channels=4096,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096,4096,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096,num_class,kernel_size=1),
        )

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self._init_weight()

    def forward(self, features):
    
        pool3 = features['layer2']
        pool4 = features['layer3']
        pool5 = features['layer4']

        pool5 = self.classifier3(pool5)

        # 1/32 *2 + 1/16
        pool5 = F.interpolate(pool5, scale_factor=2, mode='bilinear', align_corners=False)
        pool4 = self.project2(pool4)
        x = 0.5*pool5 + 0.5*pool4

        # 1/16 *2 + 1/8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        pool3 = self.project1(pool3)
        x = 0.5*x + 0.5*pool3

        # 1/8 * 8 
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x 

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

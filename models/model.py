import os
import torch
from torch import nn

class Unet(nn.Module):
    def __init__(self,num_classes=21): # Pascal VOC 2012
        super(Unet,self).__init__()
        
        def CBR2D(in_c,out_c,k=3,s=1,padding=1,bias=False):
            block = nn.Sequential(nn.Conv2d(in_channels=in_c,out_channels=out_c,
                                            kernel_size=k,stride=s,padding=padding,bias=bias),
                                  nn.BatchNorm2d(num_features=out_c),
                                  nn.ReLU())
            return block
        # 512 3
        self.enc1_1 = CBR2D(in_c=3,out_c=64)
        self.enc1_2 = CBR2D(in_c=64,out_c=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 256 64
        self.enc2_1 = CBR2D(in_c=64,out_c=128)
        self.enc2_2 = CBR2D(in_c=128,out_c=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 128 128
        self.enc3_1 = CBR2D(in_c=128,out_c=256)
        self.enc3_2 = CBR2D(in_c=256,out_c=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 64 256
        self.enc4_1 = CBR2D(in_c=256,out_c=512)
        self.enc4_2 = CBR2D(in_c=512,out_c=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 32 512
        self.enc5_1 = CBR2D(in_c=512,out_c=1024)
        
        self.dec5_1 = CBR2D(in_c=1024,out_c=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512,out_channels=512,
                                          kernel_size=2,stride=2,padding=0,bias=False)
        self.dec4_2 = CBR2D(in_c=2*512,out_c=512)
        self.dec4_1 = CBR2D(in_c=512,out_c=256)
        
        self.unpool3 = nn.ConvTranspose2d(in_channels=256,out_channels=256,
                                          kernel_size=2,stride=2,padding=0,bias=False)
        self.dec3_2 = CBR2D(in_c=2*256,out_c=256)
        self.dec3_1 = CBR2D(in_c=256,out_c=128)
        
        self.unpool2 = nn.ConvTranspose2d(in_channels=128,out_channels=128,
                                          kernel_size=2,stride=2,padding=0,bias=False)
        self.dec2_2 = CBR2D(in_c=2*128,out_c=128)
        self.dec2_1 = CBR2D(in_c=128,out_c=64)
        
        self.unpool1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,
                                          kernel_size=2,stride=2,padding=0,bias=False)
        self.dec1_2 = CBR2D(in_c=2*64,out_c=64)
        self.dec1_1 = CBR2D(in_c=64,out_c=64)
        
        self.fc = nn.Conv2d(in_channels=64,out_channels=num_classes,
                            kernel_size=1,stride=1,padding=0,bias=False)
    
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4,enc4_2),dim = 1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3,enc3_2),dim = 1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2,enc2_2),dim = 1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1,enc1_2),dim = 1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.fc(dec1_1)
        output={'out':x}
        return output

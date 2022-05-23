import random
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T

class RandomResize(object):
    def __init__(self,min_size,max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
    def __call__(self,input,target):
        size = random.randint(self.min_size,self.max_size)
        input = F.resize(input,size)
        target = F.resize(target,size)
        return input,target

class ConvertImageDtype(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class transforms_train(object):
    def __init__(self,base_size=512,crop_size=448,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)):
        transforms = []
        transforms.extend(
            [RandomResize(base_size),
             T.RandomHorizontalFlip(),
             T.RandomCrop(crop_size),
             T.PILToTensor(),
             ConvertImageDtype(torch.float),
             Normalize(mean,std)]
        )
        self.transforms = T.Compose(transforms)
    def __call__(self,input,target):
        return self.transforms(input,target)

class transforms_eval(object):
    def __init__(self,base_size=512,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)):
        transforms = []
        transforms.extend(
            [RandomResize(base_size),
             T.PILToTensor(),
             ConvertImageDtype(torch.float),
             Normalize(mean,std)]
        )
        self.transforms = T.Compose(transforms)
    def __call__(self,input,target):
        return self.transforms(input,target)


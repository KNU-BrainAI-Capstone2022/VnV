import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Compose

class ToTensor(object):
    def __call__(self, data):
        data['input'] = data['input'].transpose((2,0,1))
        data['target'] = data['target'].transpose((2,0,1))

        data['input'] = torch.from_numpy(data['input'])
        data['target'] = torch.from_numpy(data['target']).type(torch.long)

        return data

class RandomResize(object):
    def __init__(self,min_size,max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
    def __call__(self,data):
        size = random.randint(self.min_size,self.max_size)
        data['input'] = F.resize(data['input'],[size,size],F.InterpolationMode.NEAREST)
        data['target'] = F.resize(data['target'],[size,size],F.InterpolationMode.NEAREST)
        return data

class RandomHorizontalFlip(object):
    def __call__(self,data):
        if random.random() > 0.5:
            data['input'] = F.hflip(data['input'])
            data['target'] = F.hflip(data['target'])
        return data

class ConvertImageDtype(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, data):
        data['input'] = F.convert_image_dtype(data['input'], self.dtype)
        return data

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data['input'] = F.normalize(data['input'], mean=self.mean, std=self.std)
        return data

class transforms_train(object):
    def __init__(self,base_size=512,crop_size=448,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)):
        transforms = []
        transforms.extend(
            [ToTensor(),
             RandomResize(base_size),
             RandomHorizontalFlip(),
             ConvertImageDtype(torch.float),
             Normalize(mean,std)
            ]
        )
        self.transforms = Compose(transforms)
    def __call__(self,data):
        return self.transforms(data)

class transforms_eval(object):
    def __init__(self,base_size=512,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)):
        transforms = []
        transforms.extend(
            [ToTensor(),
             RandomResize(base_size),
             ConvertImageDtype(torch.float),
             Normalize(mean,std)]
        )
        self.transforms = Compose(transforms)
    def __call__(self,data):
        return self.transforms(data)

def get_transform(train=True):
    if train:
        return transforms_train(base_size=512,crop_size=448)
    else:
        return transforms_eval(base_size=512)
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Compose

class ToTensor(object):
    def __call__(self, data):
        data['image'] = data['image'].transpose((2,0,1))
        data['target'] = data['target'].transpose((2,0,1))

        data['image'] = torch.from_numpy(data['image'])
        data['target'] = torch.from_numpy(data['target'])

        return data

class RandomResize(object):
    def __init__(self,min_size,max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
    def __call__(self,data):
        size = random.randint(self.min_size,self.max_size)
        data['image'] = F.resize(data['image'],[size,size],F.InterpolationMode.NEAREST)
        data['target'] = F.resize(data['target'],[size,size],F.InterpolationMode.NEAREST)
        return data

class RandomCrop(object):
    def __init__(self, crop_size):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size

    def __call__(self, data):
        h, w = data['image'].shape[-2:]
        new_h, new_w = self.crop_size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        data['image'] = data['image'][:,top: top + new_h,left: left + new_w]
        data['target'] = data['target'][:,top: top + new_h,left: left + new_w]

        return data

class RandomHorizontalFlip(object):
    def __call__(self,data):
        if random.random() > 0.5:
            data['image'] = F.hflip(data['image'])
            data['target'] = F.hflip(data['target'])
        return data

class ConvertImageDtype(object):
    def __init__(self, dtype:list):
        self.dtype = dtype

    def __call__(self, data):
        data['image'] = F.convert_image_dtype(data['image'], self.dtype)
        return data

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data['image'] = F.normalize(data['image'], mean=self.mean, std=self.std)
        return data

class Squeeze(object):
    def __call__(self, data):
        data['target'] = data['target'].squeeze() # 1xHxW -> HxW
        return data

class transforms_train(object):
    def __init__(self,base_size=256,crop_size=224,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)):
        transforms = []
        transforms.extend(
            [ToTensor(),
             RandomResize(base_size),
             RandomCrop(crop_size),
             RandomHorizontalFlip(),
             ConvertImageDtype(torch.float),
             Normalize(mean,std),
             Squeeze(),
            ]
        )
        self.transforms = Compose(transforms)
    def __call__(self,data):
        return self.transforms(data)

class transforms_eval(object):
    def __init__(self,base_size=256,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)):
        transforms = []
        transforms.extend(
            [ToTensor(),
             RandomResize(base_size),
             ConvertImageDtype(torch.float),
             Normalize(mean,std),
             Squeeze(),
             ]
        )
        self.transforms = Compose(transforms)
    def __call__(self,data):
        return self.transforms(data)

def get_transform(train=True,base_size=512,crop_size=448):
    if train:
        return transforms_train(base_size=base_size,crop_size=crop_size)
    else:
        return transforms_eval(base_size=base_size)

if __name__ == "__main__":
    import numpy as np
    a = np.random.rand(512,512,3)
    b = np.random.randint(0,21,(512,512,1))
    print(a.shape)
    print(b.shape)
    data = {'image':a,'target':b}
    transform = transforms_train()
    data = transform(data)
    print(data['image'].shape)
    print(data['target'].shape)
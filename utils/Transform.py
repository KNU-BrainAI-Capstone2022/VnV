import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Compose
import numpy as np
from PIL import Image

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, data):
       
        data['image'] = F.to_tensor(data['image'])

        data['target'] = torch.from_numpy(data['target'].transpose((2,0,1)))

        return data

class Resize(object):
    def __init__(self,base_size, interpolation=Image.BILINEAR):
        assert isinstance(base_size, (int, tuple))

        if isinstance(base_size, int):
            self.base_size = (base_size, base_size)
        else:
            assert len(base_size) == 2
            self.base_size = base_size
        
        self.interpolation = interpolation
    
    def __call__(self, data):

        data['image'] = F.resize(data['image'],self.base_size,self.interpolation)
        data['target'] = F.resize(data['target'],self.base_size,self.interpolation)
        return data

class RandomCrop(object):
    def __init__(self, crop_size, pad_if_needed=False, padding=0):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    def __call__(self, data):
        img = data['image']
        lbl = data['target']
        
        assert img.shape[-2:] == lbl.shape[-2:], 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)
        
        # pad the width if needed
        if self.pad_if_needed and img.size(1) < self.crop_size[1]:
            img = F.pad(img, padding=int((1+ self.crop_size[1] - img.size(1)) / 2))
            lbl = F.pad(lbl, padding=int((1+ self.crop_size[1] - lbl.size(1)) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size(2) < self.crop_size[0]:
            img = F.pad(img, padding=int((1+self.crop_size[0] - img.size(2)) / 2 ))
            lbl = F.pad(lbl, padding=int((1+self.crop_size[0] - lbl.size(2)) / 2 ))

        h, w = img.shape[-2:]
        new_h, new_w = self.crop_size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        data['image'] = F.crop(img,top,left,h,w)
        data['target'] = F.crop(lbl,top,left,h,w)

        return data

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,data):
        if type(data['image']) == 'numpy.ndarray':
            data['image']=Image.fromarray(data['image'])
            data['traget'] = Image.fromarray(data['target'])
        
        if random.random() > 0.5:
            data['image'] = F.hflip(data['image'])
            data['target'] = F.hflip(data['target'])
        return data

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data['image'] = F.normalize(data['image'], mean=self.mean, std=self.std)
        return data

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
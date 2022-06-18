import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Compose
import numpy as np
from PIL import Image
import numbers

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, data):
       
        data['image'] = F.to_tensor(data['image'])

        data['target'] = torch.from_numpy(np.array(data['target'],dtype=np.uint8))

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
        data['target'] = F.resize(data['target'],self.base_size, Image.NEAREST)
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
        
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)
        
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.crop_size[1]:
            img = F.pad(img, padding=int((1+ self.crop_size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1+ self.crop_size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.crop_size[0]:
            img = F.pad(img, padding=int((1+self.crop_size[0] - img.size[1]) / 2 ))
            lbl = F.pad(lbl, padding=int((1+self.crop_size[0] - lbl.size[1]) / 2 ))

        h, w = img.size
        new_h, new_w = self.crop_size
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        data['image'] = F.crop(img,top,left,new_h,new_w)
        data['target'] = F.crop(lbl,top,left,new_h,new_w)

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

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, data):
        """
        Args:
            img (Tensor): Input image.
        Returns:
            Tensor Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        data['image'] = transform(data['image'])
        return data

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

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
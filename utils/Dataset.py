#%%
import os
import numpy as np
from PIL import Image
import torch
from collections import namedtuple
from torchvision.transforms import Compose
from Transform import *

class CustomVOCSegmentation(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_set="train", transform=None):
        self._classes = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv/monitor",
        ]

        self._cmap = [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    
        self.data_dir = data_dir
        self.image_set = image_set
        self.transform = transform

        file = os.path.join(data_dir,'VOCdevkit','VOC2012','ImageSets','Segmentation',image_set+'.txt')
        with open(file,'r') as f:
            self.list_file = f.read().split()
        self.path_jpeg = os.path.join(data_dir,'JPEGImages')
        self.path_mask = os.path.join(data_dir,'SegmentationClass')

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path_jpeg,self.list_file[index]+'.jpg')).convert('RGB')
        target = Image.open(os.path.join(self.path_mask,self.list_file[index]+'.png'))

        image = np.array(image)
        target = np.array(target)
        
        if target.ndim == 2: # HxW -> CxHxW
            target = np.expand_dims(target,axis=-1)

        data = {'image':image,'target':target}
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def getclasses(self):
        return self._classes

    def getcmap(self):
        return self._cmap

class CustomCityscapesSegmentation(torch.utils.data.Dataset):
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    def __init__(self, data_dir, image_set="train", transform=None):
        self._ignore_index = [255]
        
        self.data_dir = data_dir
        self.image_set = image_set
        self.transform = transform

        self.path_jpeg = os.path.join(data_dir,'leftImg8bit',image_set)
        self.path_mask = os.path.join(data_dir,'gtFine',image_set)
        self.images = []
        self.targets = []

        for city in os.listdir(self.path_jpeg):
            img_dir = os.path.join(self.path_jpeg,city)
            target_dir = os.path.join(self.path_mask,city)
            for file_name in os.listdir(img_dir):
                target_name ='{}_{}'.format(file_name.split('_leftImg8bit')[0],'gtFine_labelIds.png')
                self.images.append(os.path.join(img_dir,file_name))
                self.targets.append(os.path.join(target_dir,target_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        image = np.array(image, dtype=np.float32)
        target = np.array(target, dtype=np.uint8)
        
        if target.ndim == 2: # HxW -> CxHxW
            target = np.expand_dims(target,axis=-1)
        
        for l in self.classes:
            idx = target==l.id
            target[idx] = l.train_id

        data = {'image':image,'target':target}
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def getclasses(self):
        classes = []
        for i in self.classes:
            if i.train_id >=0 and i.train_id <19:
                classes.append(i.name)
        return classes

    def getcmap(self):
        cmap = []
        for i in self.classes:
            if i.train_id >=0 and i.train_id <19:
                cmap.append(i.color)
                #print(f'train id : {i.train_id} {cmap[-1]}')
        return cmap
    
def get_dataset(dir_path,dataset):

    if dataset =='voc2012':
        train_transform = Compose([
            ToTensor(),
            Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            Resize(512),
            RandomCrop(crop_size=256, pad_if_needed=True),
            RandomHorizontalFlip(),
            
        ])
        val_transform = Compose([
            ToTensor(),
            Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            Resize(512),
        ])

        train_dataset = CustomVOCSegmentation(data_dir=dir_path,
                                            image_set="train",
                                            transform=train_transform)
        val_dataset = CustomVOCSegmentation(data_dir=dir_path,
                                            image_set="val",
                                            transform=val_transform)

    if dataset =='cityscapes':
        train_transform = Compose([
            ToTensor(),
            Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            RandomCrop(512),
            RandomHorizontalFlip(),
        ])
        val_transform = Compose([
            ToTensor(),
            Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        
        train_dataset = CustomCityscapesSegmentation(data_dir=dir_path,
                                            image_set="train",
                                            transform=train_transform)
        val_dataset = CustomCityscapesSegmentation(data_dir=dir_path,
                                            image_set="val",
                                            transform=val_transform)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision
    from torch.utils.data import DataLoader
    import argparse

    def get_args():
        parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
        parser.add_argument("--dataset", choices=['voc2012','cityscapes'], type=str, help="dataset name",required=True)
        return parser.parse_args()

    def decode_segmap(masks,colormap):
        r_mask = torch.zeros_like(masks,dtype=torch.uint8)
        g_mask = torch.zeros_like(masks,dtype=torch.uint8)
        b_mask = torch.zeros_like(masks,dtype=torch.uint8)
        for k in range(len(colormap)):
            indices = masks == k
            #print(r_mask[indices].shape)
            r_mask[indices] = colormap[k][0]
            g_mask[indices] = colormap[k][1]
            b_mask[indices] = colormap[k][2]
        return torch.cat([r_mask,g_mask,b_mask],dim=1)

    kargs = vars(get_args())

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # download_pascalvoc(os.path.join(root_dir,'dataset'))

    data_dir = os.path.join(root_dir,'dataset')

    train_dataset, val_dataset = get_dataset(data_dir,kargs['dataset'])
    colormap = train_dataset.getcmap()

    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    for data in train_loader:
        images,targets = data['image'],data['target']
        fig, ax = plt.subplots(2,1,figsize=(12,6))
        ax[0].imshow(torchvision.utils.make_grid(images.cpu(), normalize=True).permute(1,2,0))
        ax[0].set_title("Input")
        ax[0].axis('off')
        #print(targets.shape)
        targets = decode_segmap(targets,colormap)
        #print(targets.shape)
        ax[1].imshow(torchvision.utils.make_grid(targets.cpu()).permute(1,2,0))
        ax[1].set_title("Target")
        ax[1].axis('off')
        
        fig.tight_layout()
        plt.show()
        break

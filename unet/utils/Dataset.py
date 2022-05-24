#%%
import os
from turtle import down
import numpy as np
from PIL import Image
import torch

class CustomVOCSegmentation(torch.utils.data.Dataset):
    classes = [
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


    colormap = [
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
    
    def __init__(self, data_dir, image_set="train", transform=None):
        self.data_dir = data_dir
        self.image_set = image_set
        self.transform = transform

        file = os.path.join(data_dir,'ImageSets','Segmentation',image_set+'.txt')
        with open(file,'r') as f:
            self.list_file = f.read().split()
        self.path_jpeg = os.path.join(data_dir,'JPEGImages')
        self.path_mask = os.path.join(data_dir,'SegmentationClass')

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, index):
        input = Image.open(os.path.join(self.path_jpeg,self.list_file[index]+'.jpg')).convert('RGB')
        target = Image.open(os.path.join(self.path_mask,self.list_file[index]+'.png'))

        input = np.array(input)
        target = np.array(target)

        if target.ndim == 2: # HxW -> CxHxW
            target = np.expand_dims(target,axis=-1)

        data = {'input':input,'target':target}
        if self.transform is not None:
            data = self.transform(data)
        return data

def download_pascalvoc(data_dir):
    from torchvision.datasets import VOCSegmentation
    _ = VOCSegmentation(root=data_dir,year="2012",image_set="trainval",download=True)

def get_dataset(dir_path,name,image_set,transform):
    paths = {
        "voc": (dir_path, CustomVOCSegmentation, 21)
    }
    p, ds_fn, num_classes = paths[name]
    ds = ds_fn(p, image_set=image_set,transform=transform)
    return ds,num_classes

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision
    from torch.utils.data import DataLoader
    from Transform import transforms_train

    def decode_segmap(masks,colormap):
        r_mask = torch.zeros_like(masks,dtype=torch.uint8)
        g_mask = torch.zeros_like(masks,dtype=torch.uint8)
        b_mask = torch.zeros_like(masks,dtype=torch.uint8)
        for k in range(len(colormap)):
            indices = masks == k
            print(r_mask[indices].shape)
            r_mask[indices] = colormap[k][0]
            g_mask[indices] = colormap[k][1]
            b_mask[indices] = colormap[k][2]
        return torch.cat([r_mask,g_mask,b_mask],dim=1)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # download_pascalvoc(os.path.join(root_dir,'dataset'))
    data_dir = os.path.join(root_dir,'dataset','VOCdevkit','VOC2012')
    
    train_dataset = CustomVOCSegmentation(data_dir=data_dir,
                                          image_set="train",
                                          transform=transforms_train())
    colormap = CustomVOCSegmentation.colormap
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    for data in train_loader:
        images,targets = data['input'],data['target']
        fig, ax = plt.subplots(2,1,figsize=(12,6))
        ax[0].imshow(torchvision.utils.make_grid(images.cpu(), normalize=True).permute(1,2,0))
        ax[0].set_title("Input")
        ax[0].axis('off')
        print(targets.shape)
        targets = decode_segmap(targets,colormap)
        print(targets.shape)
        ax[1].imshow(torchvision.utils.make_grid(targets.cpu()).permute(1,2,0))
        ax[1].set_title("Target")
        ax[1].axis('off')
        
        fig.tight_layout()
        plt.show()
        break
        
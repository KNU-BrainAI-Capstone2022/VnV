import os
from matplotlib import transforms
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import VOCSegmentation

class CustomVOCSegmentation(VOCSegmentation):
    def __init__(self, root, image_set="train", download=True, transforms=None):
        super().__init__(root=root, image_set=image_set, download=download, transforms=transforms)
        self.voc_classes = [
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
        self.voc_colormap = [
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

    def _convert_to_segmentation_mask(self,mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(self.voc_classes)), dtype=np.float32)
        for label_index, label in enumerate(self.voc_colormap):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(np.float32)
        return torch.from_numpy(segmentation_mask.transpose(2,0,1))
    
    def __getitem__(self, idx):
        input = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.masks[idx])
        
        data = (input,target)
        if self.transforms:
            data = self.transforms(data)

        return data

if __name__=="__main__":
    import os
    import torchvision
    from torch.utils.data import DataLoader
    from transforms import transforms_train
    import matplotlib.pyplot as plt
    
    def get_dataset(dir_path,name,image_set,transforms):
        def sbd(*args, **kwargs):
            return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)
        paths = {
            "voc2012": (dir_path, CustomVOCSegmentation, 21),
            "voc_aug": (dir_path, sbd, 21),
        }
        p, ds_fn, num_classes = paths[name]
        ds = ds_fn(os.path.join(p,name), image_set=image_set,download=True,transforms=transforms)
        return ds,num_classes

    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir,"dataset")
    batch_size = 16
    train_ds, num_classes = get_dataset(data_dir,"voc2012","train",transforms=transforms_train())
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
        )
    loader = iter(train_loader)
    inputs,labels = loader.next()
    print(inputs.size(),inputs.dtype())
    print(labels.size(),labels.dtype())
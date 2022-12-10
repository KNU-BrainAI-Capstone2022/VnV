voc = [
    ['background', [0, 0, 0]],
    ['aeroplane', [128, 0, 0]],
    ['bicycle', [119, 11, 32]],
    ['bird', [128, 128, 0]],
    ['boat', [0, 0, 128]],
    ['bottle', [128, 0, 128]],
    ['bus', [0, 60, 100]],
    ['car', [0, 0, 142]],
    ['cat', [64, 0, 0]],
    ['chair', [70, 70, 70]],
    ['cow', [64, 128, 0]],
    ['diningtable', [192, 128, 0]],
    ['dog', [64, 0, 128]],
    ['horse', [192, 0, 128]],
    ['motorbike', [0, 0, 230]],
    ['person', [220, 20, 60]],
    ['potted plant', [0, 64, 0]],
    ['sheep', [128, 64, 0]],
    ['sofa', [0, 192, 0]],
    ['train', [0, 80, 100]],
    ['tv/monitor', [0, 64, 128]],
]

cmap_voc = [[0, 0, 0], [128, 0, 0], [119, 11, 32], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 60, 100], [0, 0, 142], [64, 0, 0], [70, 70, 70], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [0, 0, 230], [220, 20, 60], [0, 64, 0], [128, 64, 0], [0, 192, 0], [0, 80, 100], [0, 64, 128]]

cityscapes = [
    ['road', (128, 64, 128)],
    ['sidewalk', (244, 35, 232)],
    ['building', (70, 70, 70)],
    ['wall', (102, 102, 156)],
    ['fence', (190, 153, 153)],
    ['pole', (153, 153, 153)],
    ['traffic light', (250, 170, 30)],
    ['traffic sign', (220, 220, 0)],
    ['vegetation', (107, 142, 35)],
    ['terrain', (152, 251, 152)],
    ['sky', (70, 130, 180)],
    ['person', (220, 20, 60)],
    ['rider', (255, 0, 0)],
    ['car', (0, 0, 142)],
    ['truck', (0, 0, 70)],
    ['bus', (0, 60, 100)],
    ['train', (0, 80, 100)],
    ['motorcycle', (0, 0, 230)],
    ['bicycle', (119, 11, 32)],
]

cmap_cityscapes = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

import numpy as np
def mask_colorize(masks,cmap):
    """
    Args:
        img (np.ndarray): np.ndarray of shape HxWxC and dtype uint8 (BGR Image)
        mask (np.ndarray): np.ndarray of shape HxW and dtype int range of [0,num_classes).
        cmap (np.ndarray) : np.ndarray containing the colors of shape NUM_CLASSES x C and its dtype uint8 (order : RGB)
    Returns:
        newimg (np.ndarray[CxHxW]): BGR color masking image of shape HxWxC.
    """
    assert masks.ndim == 2
    r_mask = np.zeros_like(masks,dtype=np.uint8)
    g_mask = np.zeros_like(masks,dtype=np.uint8)
    b_mask = np.zeros_like(masks,dtype=np.uint8)
    for k in range(len(cmap)):
        indices = masks == k
        r_mask[indices] = cmap[k,0]
        g_mask[indices] = cmap[k,1]
        b_mask[indices] = cmap[k,2]
    return np.stack([b_mask,g_mask,r_mask],axis=2)
# Copyright 2021 Min Seok (Karel) Lee
import skimage
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset
import torch

def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        return None
    else:
        ext = allfiles[0].split('.')[-1]
        filelist = [
            fname.replace(''.join(['.', ext]), '') for fname in allfiles
        ]
        return ext, filelist


class SalData(Dataset):
    """Dataset for saliency detection"""

    def __init__(self, dataDir, size, augmentation=True, mode='train'):
        super(SalData, self).__init__()
        if not os.path.isdir(os.path.join(dataDir, 'images')):
            raise ValueError(
                'Please put your images in folder \'images\' and GT in \'GT\'')
        self.dataDir = dataDir
        _, self.imgList = fold_files(os.path.join(dataDir, 'images'))
        self.augmentation = augmentation
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.size = size
        self.mode = mode

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgName = self.imgList[idx]
        # print(imgName)
        img = skimage.img_as_float(
            io.imread(os.path.join(self.dataDir, 'images', imgName + '.jpg')))
        gt = skimage.img_as_float(
            io.imread(os.path.join(self.dataDir, 'GT', imgName + '.png'),
                      as_gray=True))
        imgsize = gt.shape
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            img = np.repeat(img, 3, 2)
        # if self.augmentation is True and self.mode == 'train':
        #     aug = Augment(size_h=15, size_w=15, p_flip=0.5)
        #     img, gt = aug(img, gt)
        img = resize(img, (self.size[0], self.size[1]),
                     mode='reflect',
                     anti_aliasing=False)
        if self.mode == 'train':
            gt = resize(gt, (self.size[0], self.size[1]),
                        mode='reflect',
                        anti_aliasing=False)
        # Normalize image
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        gt = gt[np.newaxis, ::]
        if self.mode == "train":
            sample = {'img': img, 'gt': gt}
        else:
            sample = {'img': img, 'gt': gt, 'h': imgsize[0], 'w': imgsize[1]}
        return sample


def val_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    data = torch.stack([torch.from_numpy(item['img']) for item in batch], 0)
    target = [torch.from_numpy(item['gt']) for item in batch]
    h = [item['h'] for item in batch]
    w = [item['w'] for item in batch]
    # target = torch.LongTensor(target)
    return [data, target, h, w]
    raise TypeError((error_msg.format(type(batch[0]))))

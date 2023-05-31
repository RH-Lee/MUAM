# Copyright 2021 Min Seok (Karel) Lee
import os
import random
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2


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


class Augment(object):
    """
    Augment image as well as target(image like array, not box)
    augmentation include Crop Pad and Filp
    """

    def __init__(self, size_h=15, size_w=15, padding=None, p_flip=None):
        super(Augment, self).__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.padding = padding
        self.p_flip = p_flip

    def get_params(self, img):
        im_sz = img.shape[:2]
        row1 = random.randrange(self.size_h)
        row2 = -random.randrange(
            self.size_h) - 1  # minus 1 to avoid row1==row2==0
        col1 = random.randrange(self.size_w)
        col2 = -random.randrange(self.size_w) - 1
        if row1 - row2 >= im_sz[0] or col1 - col2 >= im_sz[1]:
            raise ValueError(
                "Image size too small, please choose smaller crop size")
        padding = None
        if self.padding is not None:
            padding = random.randint(0, self.padding)
        flip_method = None
        if self.p_flip is not None and random.random() < self.p_flip:
            if random.random() < 0.5:
                flip_method = 'lr'
            else:
                flip_method = 'ud'
        return row1, row2, col1, col2, flip_method, padding

    def transform(self,
                  img,
                  row1,
                  row2,
                  col1,
                  col2,
                  flip_method,
                  padding=None):
        """img should be 2 or 3 dimensional numpy array"""
        img = img[row1:row2,
              col1:col2, :] if len(img.shape) == 3 else img[row1:row2,
                                                        col1:col2]
        if padding is not None:  # TODO: not working yet, fix it later
            pad = transforms.Pad(padding)
            topil = transforms.ToPILImage()
            img = pad(topil(img))
            img = np.array(img)
        if flip_method is not None:
            if flip_method == 'lr':
                img = np.fliplr(img)
            else:
                img = np.flipud(img)
        return img

    def __call__(self, img, target):
        """img and target should have the same spatial size"""
        paras = self.get_params(img)
        img = self.transform(img, *paras)
        target = self.transform(target, *paras)
        return img, target


class SalData(Dataset):
    """Dataset for saliency detection"""

    def __init__(self, dataDir, size, transform, augmentation=True, mode='train'):
        super(SalData, self).__init__()
        if not os.path.isdir(os.path.join(dataDir, 'images')):
            raise ValueError(
                'Please put your images in folder \'images\' and GT in \'GT\'')
        self.dataDir = dataDir
        _, self.imgList = fold_files(os.path.join(dataDir, 'images'))
        self.augmentation = augmentation
        self.transform = transform
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.size = size
        self.mode = mode

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgName = self.imgList[idx]

        image = cv2.imread(os.path.join(self.dataDir, 'images', imgName + '.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        edge = cv2.imread(os.path.join(self.dataDir, 'edge', imgName + '_edge.png'))
        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(os.path.join(self.dataDir, 'GT', imgName + '.png'))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            augmented = self.transform(image=image, masks=[mask, edge])
            image = augmented['image']
            mask = np.expand_dims(augmented['masks'][0], axis=0)  # (1, H, W)
            mask = mask / 255.0
            edge = np.expand_dims(augmented['masks'][1], axis=0)  # (1, H, W)
            edge = edge / 255.0
        sample = {'img': image, 'gt': mask, 'edge': edge}
        return sample


class pSalData(Dataset):
    """Dataset for saliency detection"""

    def __init__(self, dataDir, size, transform, augmentation=True):
        super(pSalData, self).__init__()
        if not os.path.isdir(os.path.join(dataDir, 'images')):
            raise ValueError(
                'Please put your images in folder \'images\' and GT in \'GT\'')
        self.dataDir = os.path.join(dataDir, 'images')
        self.images = sorted(glob.glob(self.dataDir + '/*'))
        self.gtDir = os.path.join(dataDir, 'GT')
        self.gts = sorted(glob.glob(self.gtDir + '/*'))
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.gts[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255.0
        mask = np.where(mask > 0.5, 1.0, 0.0)

        imgsize = mask.shape

        # if self.transform is not None:
        augmented = self.transform(image=image)
        image = augmented['image']
        image = np.transpose(image, (2, 0, 1))
        mask = mask[np.newaxis, ::]
        # print(image.shape)    # (3, 224, 224)

        sample = {'img': image, 'gt': mask, 'h': imgsize[0], 'w': imgsize[1]}
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

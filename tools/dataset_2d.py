"""
This part is based on the dataset class implemented by pytorch, 
including train_dataset and test_dataset, as well as data augmentation
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize


class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob <= self.prob:
            img = np.flip(img,1)
        return img

    def __call__(self, node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge):
        prob = random.uniform(0, 1)
        return self._flip(node_img, prob), self._flip(node_skl, prob), self._flip(node_skl_dis, prob), self._flip(node_walk, prob), self._flip(node_point, prob), self._flip(node_edge, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob <= self.prob:
            img = np.flip(img,0)
        return img

    def __call__(self, node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge):
        prob = random.uniform(0, 1)
        return self._flip(node_img, prob), self._flip(node_skl, prob), self._flip(node_skl_dis, prob), self._flip(node_walk, prob), self._flip(node_point, prob), self._flip(node_edge, prob)

class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = np.rot90(img, cnt, [1,2])
        return img

    def __call__(self, node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge):
        cnt = random.randint(0,self.max_cnt)
        return self._rotate(node_img, cnt), self._rotate(node_skl, cnt), self._rotate(node_skl_dis, cnt), self._rotate(node_walk, cnt), self._rotate(node_point, cnt), self._rotate(node_edge, cnt)

class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge):
        node_img = torch.from_numpy(node_img).float()#.unsqueeze(0)
        node_skl = torch.from_numpy(node_skl).float()#.unsqueeze(0)
        node_skl_dis = torch.from_numpy(node_skl_dis).float()#.unsqueeze(0)
        node_walk = torch.from_numpy(node_walk).float()#.unsqueeze(0)
        node_point = torch.from_numpy(node_point).float()#.unsqueeze(0)
        node_edge = torch.from_numpy(node_edge).float()#.unsqueeze(0)

        return node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge



class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge):
        for t in self.transforms:
            node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge = t(node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge)
        return node_img, node_skl, node_skl_dis, node_walk, node_point, node_edge

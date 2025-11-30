from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import random
import numpy as np
from lib.klib.baseio import *
import copy
# from tools.Image_Tools import gen_circle_2d, gen_circle_gaussian_2d
import time
import pandas as pd
from tools.dataset_2d import RandomFlip_LR, RandomFlip_UD, RandomRotate, Compose, ToTensor

from configs import config_2d

args = config_2d.args
resize_radio = args.resize_radio
r_resize = args.r_resize
data_shape = args.data_shape


class Images_Dataset_folder_2d(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir, images_dir_list, labels_dir_list, mode):
        self.image_names = sorted(images_dir_list)
        self.labels = sorted(labels_dir_list)
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        if mode == 'train':
            self.tx_total = Compose([
            
            # RandomFlip_LR(prob=0.5),
            # RandomFlip_UD(prob=0.5),
            # RandomRotate(),
            ToTensor()
            ])
        else:
            self.tx_total = Compose([
            ToTensor()
            ])

            

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, i):

        begin_time = time.time()
        i_img = open_tif(self.images_dir + self.image_names[i] + '/' + 'node_img.tif').astype(np.float32)

        i_img1 = copy.deepcopy(i_img[0]) / 255
        i_new = i_img1.transpose((2,0,1)).astype(np.float32)
        

        i_skl = open_tif(self.images_dir + self.image_names[i] + '/' + 'node_skl.tif').astype(np.float32)
        i_skl1 = copy.deepcopy(i_skl) / 255
        i_new_skl = i_skl1.reshape(1, data_shape[0], data_shape[1]).astype(np.float32)
        

        i_walk = open_tif(self.images_dir + self.image_names[i] + '/' + 'node_walk.tif').astype(np.float32) 
        i_walk1 = copy.deepcopy(i_walk) / 255
        i_new_walk = i_walk1.reshape(1, data_shape[0], data_shape[1]).astype(np.float32)

        
        i_point = open_tif(self.images_dir + self.image_names[i] + '/' + 'node_point.tif').astype(np.float32)
        th = 128
        i_point[i_point<th]=0
        i_point[i_point>=th]=255
        i_point1 = copy.deepcopy(i_point) / 255
        i_new_point = i_point1.reshape(1, data_shape[0], data_shape[1]).astype(np.float32)
        
        i_edge = open_tif(self.images_dir + self.image_names[i] + '/' + 'node_edge.tif').astype(np.float32) 
        i_edge1 = copy.deepcopy(i_edge) / 255 # 标准为0-1
        i_new_edge = i_edge1.reshape(1, data_shape[0], data_shape[1]).astype(np.float32)
        
        i_skl_b = open_tif(self.images_dir + self.image_names[i] + '/' + 'node_skl_b.tif').astype(np.float32) 

        i_new_skl_b = copy.deepcopy(i_skl_b) / 255
        i_new_skl_b = i_new_skl_b.reshape(1, data_shape[0], data_shape[1]).astype(np.float32)
        
        node_exist = np.zeros([1])
        if self.image_names[i].split("/")[1].split("_")[1] == 'pos':
            node_exist[0] = 1
        elif self.image_names[i].split("/")[1].split("_")[1] == 'neg':
            node_exist[0] = 0
        else:
            print("error")
            pause


        seed = np.random.randint(0, 2**32) # make a seed with numpy generator
        node_img, node_skl, node_skl_b, node_walk, node_point, node_edge = self.tx_total(i_new, i_new_skl, i_new_skl_b, i_new_walk, i_new_point, i_new_edge)


        
        return node_img, node_skl, node_skl_b, node_walk, node_point, node_edge, node_exist


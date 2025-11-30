from __future__ import print_function, division
import os
from os import path
import numpy as np
from PIL import Image
import glob
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F
import torch.nn
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchsummary
from torchstat import stat
from torchsummary import summary
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import natsort
from tensorboardX import SummaryWriter
from thop import profile
from rtree import index

import shutil
import random
import pickle

# import dill


from models.RPNet_3D import PE_Net_3D
from tools.tracing.tracing_tools_3D import get_pos_image_3d, get_network_predict_vecroad_3d, tracing_strategy_single_vecroad_3d, tracing_strategy_single_vecroad_3d_test
from tools.Data_Loader_3d import Images_Dataset_folder_3d
from tools.Losses import dice_loss, MSE_loss, L1_loss, dice_score, bce_loss_w

from skimage import morphology, transform

import time

from configs import config_3d
from lib.klib.baseio import *
from lib.swclib.swc_io import swc_save_metric, swc_save, read_swc_tree
from lib.swclib.swc_tree import SwcTree
from lib.swclib.swc_node import SwcNode
from lib.swclib.euclidean_point import EuclideanPoint
from lib.swclib.edge_match_utils import get_bounds
from lib.swclib import edge_match_utils
from lib.swclib.re_sample import up_sample_swc_tree

import copy


args = config_3d.args
resize_radio = args.resize_radio
r_resize = args.r_resize


def train(args, model_name, device_ids, device):
    
    #######################################################
    #     Setting the basic paramters of the model
    #######################################################
    batch_size = args.batch_size
    valid_size = args.valid_rate
    epoch = args.epochs
    initial_lr = args.lr
    num_workers = args.n_threads
    data_shape = args.data_shape

    to_restore = args.to_restore
    hidden_layer_size = args.hidden_layer_size
    vector_bins = args.vector_bins


    lambda_centerline = 1
    lambda_point = 1
    lambda_edge = 1
    lambda_edge_topo = 1



    random_seed = random.randint(1, 100)
    shuffle = True
    lossT = []
    lossL = []
    lossL.append(np.inf)
    lossT.append(np.inf)
    epoch_valid = epoch-2
    i_valid = 0

    train_on_gpu = torch.cuda.is_available()
    pin_memory = False
    if train_on_gpu:
        pin_memory = True
    #######################################################
    #               load the data
    #######################################################
    print('loading the train data')
    train_data_dir = args.dataset_img_path
    train_label_dir = args.dataset_img_path
    
    train_data_dir_list = []
    train_label_dir_list = []

    train_data_imagename_dir_list = os.listdir(train_data_dir)

    ssss = 0
    for imagename in train_data_imagename_dir_list:
        train_data_imagename_num_dir_list = os.listdir(train_data_dir + imagename)
        for imagename_num in train_data_imagename_num_dir_list:
            train_data_dir_list.append(imagename + '/' + imagename_num)
            train_label_dir_list.append(imagename + '/' + imagename_num)
            ssss += 1

    Training_Data = Images_Dataset_folder_3d(train_data_dir, train_label_dir, train_data_dir_list, train_label_dir_list, datasets_plag)

    num_train = len(Training_Data)

    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))


    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,num_workers=num_workers, pin_memory=pin_memory,)
    valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,num_workers=num_workers, pin_memory=pin_memory,)

    #######################################################
    #               load the test data
    #######################################################
    print('loading the test data')
    train_data_dir = args.dataset_img_test_path
    train_label_dir = args.dataset_img_test_path
    
    train_data_dir_list = []
    train_label_dir_list = []

    train_data_imagename_dir_list = os.listdir(train_data_dir)

    for imagename in train_data_imagename_dir_list:
        train_data_imagename_num_dir_list = os.listdir(train_data_dir + imagename)
        for imagename_num in train_data_imagename_num_dir_list:
            train_data_dir_list.append(imagename + '/' + imagename_num)
            train_label_dir_list.append(imagename + '/' + imagename_num)


    Training_Data = Images_Dataset_folder_3d(train_data_dir, train_label_dir, train_data_dir_list, train_label_dir_list, mode = 'test')

    num_train = len(Training_Data)

    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.1 * num_train))

    test_idx = indices[split:]
    test_sampler = SubsetRandomSampler(test_idx)
    
    test_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=test_sampler,num_workers=num_workers, pin_memory=pin_memory,)

    #######################################################
    #               build the model
    #######################################################
    def model_unet(model_input, in_channel, out_channel):
        model = model_input(in_channel, out_channel)
        return model

    model_train = model_unet(model_name, 1, 1, freeze_plag)
    model_train = torch.nn.DataParallel(model_train, device_ids=device_ids)
    model_train.to(device)

    softmax = torch.nn.Softmax()
    criterion = torch.nn.CrossEntropyLoss()
    reg_criterion = torch.nn.MSELoss()
    bce_criterion = torch.nn.BCELoss()
    sigmoid_layer = torch.nn.Sigmoid()

    
    NLL_criterion = torch.nn.NLLLoss()


    opt = torch.optim.Adam(model_train.parameters(), lr=initial_lr) # try SGD
    MAX_STEP = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=0)
    
    #######################################################
    #               model and log dir
    #######################################################
    LOG_DIR = str(args.log_save_dir) + str(args.gpu_id) + '/'
    MODEL_DIR = str(args.model_save_dir) + str(args.gpu_id) + '/'


    try:
        shutil.rmtree(LOG_DIR)
        print('Model folder there, so deleted for newer one')
        os.mkdir(LOG_DIR)
    except OSError:
        print("Creation of the log directory '%s' failed " % LOG_DIR)
    else:
        print("Successfully created the log directory '%s' " % LOG_DIR)
    writer = SummaryWriter(LOG_DIR)

    try:
        os.mkdir(MODEL_DIR)
    except OSError:
        print("Creation of the model directory '%s' failed " % MODEL_DIR)
    else:
        print("Successfully created the model directory '%s' " % MODEL_DIR)
    
    

    if to_restore:
        print("loading model")
        model_train.load_state_dict(torch.load(MODEL_DIR   + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth'))
    else:
        read_model_path = MODEL_DIR  + str(epoch) + '_' + str(batch_size)
        print(read_model_path)
        if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
            shutil.rmtree(read_model_path)
            print('Model folder there, so deleted for newer one')
        try:
            os.mkdir(read_model_path)
        except OSError:
            print("Creation of the model directory '%s' failed" % read_model_path)
        else:
            print("Successfully created the model directory '%s' " % read_model_path)

    ######################DATA NROM########################

    total_step = train_loader.__len__()

    # pause
    idx_tensor = [(idx) for idx in range(vector_bins)]
    idx_tensor = torch.autograd.Variable(torch.FloatTensor(idx_tensor)).to(device)
    #=============================================================================
    print("begin training")


    valid_loss_min = np.Inf
    r_loss_min = np.Inf

    n_iter = 1
    global_step_ = 0
    for i in range(epoch):
        train_loss = 0.0
        valid_loss = 0.0
        valid_class_loss = 0.0
        valid_reg_loss = 0.0
        valid_rad_loss = 0.0
        
        since = time.time()
        scheduler.step(i)
        print('learning rate: %f' % (opt.param_groups[0]['lr']))

        train_step_temp = 0
        valid_step_temp = 0
        
        #######################################################
        #                    Training Data
        #######################################################
        model_train.train()

        load_begin_time_1 = time.time()
        for x_img, y_skl, y_skl_b, x_walk, y_point, y_edge, x_exist in train_loader:
            train_begin_time = time.time()
            
            exist_pos_id = np.where(x_exist==1)

            x_img, y_skl, y_skl_b, x_walk, y_point, y_edge = x_img.to(device), y_skl.to(device), y_skl_b.to(device), x_walk.to(device), y_point.to(device), y_edge.to(device)

            opt.zero_grad()  
            y_skl_pred, y_point_pred, y_edge_pred = model_train(x_img, x_walk)
            
            
            y_exist_skl = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1], data_shape[2]], dtype=torch.float32)
            y_edge_topo = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1], data_shape[2]], dtype=torch.float32)

            
            y_exist_point = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1], data_shape[2]], dtype=torch.float32)
            y_exist_point_pred = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1], data_shape[2]], dtype=torch.float32)
            
            y_exist_edge = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1], data_shape[2]], dtype=torch.float32)
            y_exist_edge_pred = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1], data_shape[2]], dtype=torch.float32)
            
            
            exist_num = 0
            for exist_id_num in exist_pos_id[0]:
                y_exist_skl[exist_num] = y_skl_b[exist_id_num]
                
                y_exist_point[exist_num] = y_point[exist_id_num]
                y_exist_point_pred[exist_num] = y_point_pred[exist_id_num]
                y_exist_edge[exist_num] = y_edge[exist_id_num]
                y_exist_edge_pred[exist_num] = y_edge_pred[exist_id_num]
                exist_num += 1
                

            loss_img_centerline = lambda_centerline * bce_loss_w(y_skl_pred, y_skl, 0.9)
            loss_img_point = lambda_point * bce_loss_w(y_exist_point_pred, y_exist_point, 0.9)
            loss_img_edge = lambda_edge * bce_loss_w(y_exist_edge_pred, y_exist_edge, 0.9)
            loss_img_edge_topo = lambda_edge_topo * L1loss(y_exist_edge_pred - y_exist_skl* y_exist_edge_pred, y_edge_topo)
            
            
            
            lossT =  loss_img_centerline + loss_img_edge + loss_img_point + loss_img_edge_topo
            train_loss += lossT.item()
            lossT.backward()
            opt.step()


            train_end_time = time.time()
            if (train_step_temp+1) % 20 == 0:
                lossdice_boundary = dice_score(y_lab_pred, y_lab)
                lossdice_centerline = dice_score(y_skl_pred, y_skl)
                lossdice_point = dice_score(y_exist_point_pred, y_exist_point)
                
                print('Epoch: {}/{} \t Step: {}/{} \t Total Loss: {:.5f} \t Time: {:.5f}/step'.format(i + 1, epoch, train_step_temp, total_step, lossT.item(),  train_end_time-train_begin_time))
                print('Boundary Loss: {:.5f} \t F1: {:.5f} \t | Centerline Loss: {:.5f} \t F1: {:.5f} \t | Point Loss: {:.5f} \t F1: {:.5f} \t'.format(loss_img_boundary.item(),lossdice_boundary.item(), loss_img_centerline.item(), lossdice_centerline.item(), loss_img_point.item(), lossdice_point.item()))


            train_step_temp += 1
            writer.add_scalar('Training Point Loss', loss_img_point.item(), global_step=global_step_)
            writer.add_scalar('Training Centerline Loss', loss_img_centerline.item(), global_step=global_step_)
            writer.add_scalar('Training Boundary Loss', loss_img_boundary.item(), global_step=global_step_)
            writer.add_scalar('Training Total Loss', lossT.item(), global_step=global_step_)
            global_step_ += 1

            load_begin_time_1 = time.time()
            
        #######################################################
        #               To write in Tensorboard
        #######################################################

        train_loss = train_loss / train_step_temp

        if (i+1) % 1 == 0:
            print('Epoch: {}/{} \t Training Loss: {:.6f} '.format(i + 1, epoch, train_loss))

        #######################################################
        #                    Early Stopping
        #######################################################
        

        torch.save(model_train.state_dict(), MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')

        time_elapsed = time.time() - since
        print('this epoch time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        n_iter += 1



def test_dis(args, model_name, device_ids, device):
    predict_centerline_path = args.predict_centerline_path
    if not os.path.exists(predict_centerline_path):
        os.makedirs(predict_centerline_path)
    # test_image_mask_root_dir = args.test_data_mask_path

    #######################################################
    #     Setting the basic paramters of the model
    #######################################################
    batch_size = args.batch_size
    epoch = args.epochs
    num_workers = args.n_threads
    vector_bins = args.vector_bins
    data_shape = args.data_shape

    resize_radio = args.resize_radio

    #######################################################
    #               build the model
    #######################################################
    def model_unet_LSTM(model_input):
        model = model_input()
        return model

    model_test = model_unet_LSTM(model_name)
    model_test = torch.nn.DataParallel(model_test, device_ids=device_ids)
    model_test.to(device)

    #######################################################
    #               load the checkpoint
    #######################################################
    
    MODEL_DIR = str(args.model_save_dir) + str(args.gpu_id) + '/'
    checkpoint = torch.load(MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')
    model_test.load_state_dict(checkpoint)
    model_test.eval()

    #######################################################
    #                tracing the image
    #######################################################
    data_transform = torchvision.transforms.Compose([])

    TEST_DIR = args.test_data_path
    test_image = glob.glob(TEST_DIR + '*.tif')
    print(len(test_image))

    test_patch_height = args.test_patch_height
    test_patch_width  = args.test_patch_width
    test_patch_depth  = args.test_patch_depth

    stride_height = args.stride_height
    stride_width = args.stride_width
    stride_depth = args.stride_depth

    for test_img_dir in test_image:
        image_name = test_img_dir.split('/')[-1].split('.')[0]
        img_new = open_tif(test_img_dir).astype(np.float32)
        stack_org = np.sqrt(copy.deepcopy(img_new)) / 255 


        # 重调大小
        d, h, w = stack_org.shape
        d_new = ((d-(test_patch_depth-stride_depth))//(stride_depth)+1)*(stride_depth) + (test_patch_depth-stride_depth)
        h_new = ((h-(test_patch_height-stride_height))//(stride_height)+1)*(stride_height) + (test_patch_height-stride_height)
        w_new = ((w-(test_patch_width-stride_width))//(stride_width)+1)*(stride_width) + (test_patch_width-stride_width)

        
        stack_img_new = np.zeros([d_new, h_new, w_new],dtype=np.float32)
        stack_img_new[0:d,0:h,0:w] = copy.deepcopy(stack_org)
        
        shape_lab_image = np.zeros([d_new, h_new, w_new],dtype=np.float32)
        shape_skl_image = np.zeros([d_new, h_new, w_new],dtype=np.float32)
        count_image = np.zeros([d_new, h_new, w_new],dtype=np.float32)

        shape_list = []
        SHAPE = [data_shape[0],data_shape[1],data_shape[2]]
        batch_list = []
        batch_lab_list = []
        batch_skl_list = []
        cnt = 0

        for k in range(0, d_new-(test_patch_depth-stride_depth), stride_depth):
            for j in range(0, h_new-(test_patch_height-stride_height), stride_height):
                for i in range(0, w_new-(test_patch_width-stride_width), stride_width):
                    patch = stack_img_new[k:k + SHAPE[0], j:j + SHAPE[1], i:i + SHAPE[2]]
                    r_d, r_h, r_w = patch.shape  
                    batch_list.append(patch.astype(np.float32))

        batch_matrix = np.zeros(dtype=np.float32, shape=[len(batch_list), 1, *SHAPE])

        for i in range(len(batch_list)):
            batch_matrix[i,:,:,:,:] = copy.deepcopy(batch_list[i])
        
        batch_input = data_transform(batch_matrix)
        train_loader = torch.utils.data.DataLoader(batch_input, batch_size=batch_size, num_workers=num_workers)
        
        num = train_loader.__len__()
        num_temp = 0
        for x_batch in train_loader:
            if num_temp % 10 == 0:
                print('num:%d / %d '% (num_temp,num))
            num_temp+=1
            x_img = x_batch.to(device)
            
            x_walk = torch.zeros_like(x_img)
            y_lab_pred, y_skl_pred, _ = model_test(x_img, x_walk)

            pred_lab = y_lab_pred.cpu().detach().numpy()
            pred_skl = y_skl_pred.cpu().detach().numpy()
            for i in range(pred_skl.shape[0]):
                batch_lab_list.append(pred_lab[i,0,:,:,:])
                batch_skl_list.append(pred_skl[i,0,:,:,:])
        
        num = 0
        for k in range(0, d_new-(test_patch_depth-stride_depth), stride_depth):
            for j in range(0, h_new-(test_patch_height-stride_height), stride_height):
                for i in range(0, w_new-(test_patch_width-stride_width), stride_width):
                    shape_lab_image[k:k + SHAPE[0],j:j + SHAPE[1], i:i + SHAPE[2]] += batch_lab_list[num] 
                    shape_skl_image[k:k + SHAPE[0],j:j + SHAPE[1], i:i + SHAPE[2]] += batch_skl_list[num] 
                    count_image[k:k + SHAPE[0],j:j + SHAPE[1], i:i + SHAPE[2]] += 1
                    
                    num += 1
                    
        
        shape_lab_image = shape_lab_image / count_image
        shape_skl_image = shape_skl_image / count_image
        shape_lab_image_new = (shape_lab_image[0:d,0:h,0:w])*255
        shape_skl_image_new = (shape_skl_image[0:d,0:h,0:w])*255

        file_newname_lab = predict_centerline_path + image_name + '.pro.lab.tif'
        file_newname_skl = predict_centerline_path + image_name + '.pro.skl.tif'
        save_tif(shape_lab_image_new, file_newname_lab, np.uint8)
        save_tif(shape_skl_image_new, file_newname_skl, np.uint8)


def test(args, model_name, device_ids, device):
    predict_swc_path = args.predict_swc_path
    if not os.path.exists(predict_swc_path):
        os.makedirs(predict_swc_path)
    
    predict_seed_path = args.predict_seed_path
    predict_centerline_path = args.predict_centerline_path
    # gold_seed_path = args.gold_seed_path

    #######################################################
    #     Setting the basic paramters of the model
    #######################################################
    batch_size = args.batch_size
    valid_size = args.valid_rate
    epoch = args.epochs
    initial_lr = args.lr
    num_workers = args.n_threads
    hidden_layer_size = args.hidden_layer_size
    vector_bins = args.vector_bins
    data_shape = args.data_shape

    test_patch_height = args.test_patch_height
    test_patch_width  = args.test_patch_width
    test_patch_depth  = args.test_patch_depth

    stride_height = args.stride_height
    stride_width = args.stride_width
    stride_depth = args.stride_depth


    SHAPE = [data_shape[0],data_shape[1],data_shape[2]]

    #######################################################
    #               build the model
    #######################################################
    def model_unet_LSTM(model_input):
        model = model_input()
        return model

    model_test = model_unet_LSTM(model_name)
    model_test = torch.nn.DataParallel(model_test, device_ids=device_ids)
    model_test.to(device)

    #######################################################
    #               load the checkpoint
    #######################################################
    
    MODEL_DIR = str(args.model_save_dir) + str(args.gpu_id) + '/'
    checkpoint = torch.load(MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')
    model_test.load_state_dict(checkpoint)
    model_test.eval()
    #######################################################
    #                load the data
    #######################################################
    TEST_DIR = args.test_data_path
    test_image = glob.glob(TEST_DIR + '*.tif')


    #######################################################
    #                tracing the image
    #######################################################
    torch.multiprocessing.set_start_method('spawn')

    device_info = [model_test, device]
    data_info = [SHAPE, vector_bins]



    for test_img_dir in test_image:
        image_name = test_img_dir.split('/')[-1].split('.')[0]

        stack_img = open_tif(test_img_dir).astype(np.float32)
        stack_img = transform.resize(stack_img,(round(stack_img.shape[0]*resize_radio), round(stack_img.shape[1]*resize_radio), round(stack_img.shape[2]*resize_radio)))

        
        # =========== should be replaced =====================
        
        seed_list_temp = [[],[],[]]
        stack_seed_swc_dir = predict_seed_path + image_name + '.swc' 


        seed_tree = read_swc_tree(stack_seed_swc_dir)
        seed_tree = up_sample_swc_tree(seed_tree, 5.0) 
        for tn in seed_tree.get_node_list():
            seed_list_temp[0].append(round(tn.get_z()*resize_radio)+SHAPE[0])
            seed_list_temp[1].append(round(tn.get_y()*resize_radio)+SHAPE[1])
            seed_list_temp[2].append(round(tn.get_x()*resize_radio)+SHAPE[2])
        
        indices = list(range(len(seed_list_temp[0])))
        np.random.shuffle(indices)
        seed_list = [[],[],[]]
        for i in range(len(seed_list_temp[0])):
            num = indices[i]
            seed_list[0].append(seed_list_temp[0][num])
            seed_list[1].append(seed_list_temp[1][num])
            seed_list[2].append(seed_list_temp[2][num])
            
        # load the skl
        stack_lab_dir = predict_centerline_path + image_name + '.pro.lab.tif' 
        stack_skl_dir = predict_centerline_path + image_name + '.pro.skl.tif' 

        stack_lab = open_tif(stack_lab_dir).astype(np.float32)
        stack_skl = open_tif(stack_skl_dir).astype(np.float32)

        
        th = 128
        stack_skl_b = copy.deepcopy(stack_skl)
        stack_skl_b[stack_skl<th]=0
        stack_skl_b[stack_skl>=th]=1
        
        stack_skl = morphology.skeletonize(stack_skl_b)//255


        tracing_strategy_flag = 'netracer'
        
        # ===============resize the image=====================
        d, h, w = stack_img.shape
        d_new = ((d-(test_patch_depth-stride_depth))//(stride_depth)+1)*(stride_depth) + (test_patch_depth-stride_depth)
        h_new = ((h-(test_patch_height-stride_height))//(stride_height)+1)*(stride_height) + (test_patch_height-stride_height)
        w_new = ((w-(test_patch_width-stride_width))//(stride_width)+1)*(stride_width) + (test_patch_width-stride_width)
        stack_img_new = np.zeros([d_new, h_new, w_new],dtype=np.float32)
        stack_lab_new = np.zeros([d_new, h_new, w_new],dtype=np.uint8)
        stack_skl_new = np.zeros([d_new, h_new, w_new],dtype=np.uint8)


        stack_img_new[0:d,0:h,0:w] = copy.deepcopy(stack_img)
        stack_lab_new[0:d,0:h,0:w] = copy.deepcopy(stack_lab)
        stack_skl_new[0:d,0:h,0:w] = copy.deepcopy(stack_skl)

        org_img_shape = stack_img_new.shape
        org_img_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1],org_img_shape[2]+2*SHAPE[2]])
        org_lab_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1],org_img_shape[2]+2*SHAPE[2]])
        org_skl_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1],org_img_shape[2]+2*SHAPE[2]])

        
        org_img_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1], 1*SHAPE[2]:1*SHAPE[2] + org_img_shape[2]] = copy.deepcopy(stack_img_new)
        org_lab_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1], 1*SHAPE[2]:1*SHAPE[2] + org_img_shape[2]] = copy.deepcopy(stack_lab_new)
        org_skl_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1], 1*SHAPE[2]:1*SHAPE[2] + org_img_shape[2]] = copy.deepcopy(stack_skl_new)


        shape_image = np.zeros([d_new, h_new, w_new],dtype=np.float32)
        # =====================================================


        seed_list_flag = np.ones(len(seed_list[0]))
        print('seed node number: %d' % (seed_list_flag.shape[0]))

        pos_list = []

        test_image = copy.deepcopy(org_img_temp).astype(np.uint8)

        # build tree
        tree_new = SwcTree()
        tree_new_idedge_dict = {}
        swc_tree_list = tree_new.get_node_list()
        p = index.Property()
        p.dimension = 3
        tree_new_rtree = index.Index(properties=p)
        for node in swc_tree_list:
            if node.is_virtual() or node.parent.is_virtual():
                continue
            tree_new_rtree.insert(node.get_id(), get_bounds(node, node.parent, extra=node.radius()))
            tree_new_idedge_dict[node.get_id()] = tuple([node, node.parent])    
        r_tree_info = [tree_new_rtree, tree_new_idedge_dict]


        test_id = 0

        begin_time_a = time.time()
        for i in range(seed_list_flag.shape[0]):

            begin_time = time.time()
            # print(i)
            if i == 0:
                glob_node_id = 1
            else:
                try:
                    glob_node_id = max(tree_new.id_set) + 1
                except:
                    glob_node_id = 1
            
            print("------------------- tracing ----------------------", i + 1, " / ", seed_list_flag.shape[0])

            seed_node_z = seed_list[0][i]
            seed_node_x = seed_list[1][i]
            seed_node_y = seed_list[2][i]

            
            # init seed node
            seed_node_dict = {'node_id': glob_node_id, 'node_z': seed_node_z, 'node_x': seed_node_x, 'node_y': seed_node_y, 'node_r': 2, 'node_p_id': -1, 'parent_node': tree_new._root, 'node_exist': 255, 'node_list_walked':[[seed_node_z, seed_node_x, seed_node_y]]}
            seed_node_dict['direction_vector'] = [0,0,0]

            seed_node = SwcNode(nid=seed_node_dict['node_id'], ntype=0, center=EuclideanPoint(center=[seed_node_dict['node_y'],seed_node_dict['node_x'],seed_node_dict['node_z']]), radius=round(seed_node_dict['node_r'],3))
            
            
            node_temp_list = edge_match_utils.get_nearby_edges(rtree=tree_new_rtree, point=seed_node, id_edge_dict=tree_new_idedge_dict, threshold=2)
            if len(node_temp_list) != 0:
                print("this seed is already traced")
                continue
            
            # process the seed node
            node_list_walked = seed_node_dict['node_list_walked']
            seed_node_img, seed_node_img_walked = get_pos_image_3d(org_img_temp, node_list_walked, [seed_node_z, seed_node_x, seed_node_y], SHAPE)
            seed_node_img = seed_node_img.reshape(1,*SHAPE)
            seed_node_img_walked = seed_node_img_walked.reshape(1,*SHAPE)
            exist, next_node_pos = get_network_predict_vecroad_3d(org_lab_temp, org_skl_temp, seed_node_img, node_list_walked, seed_node_img_walked, SHAPE, model_test, device, vector_bins, tracing_strategy_flag)

            
            seed_node.parent = seed_node_dict['parent_node']
            
            # start tracing
            tree_new, r_tree_info = tracing_strategy_single_vecroad_3d_test(org_img_temp, org_lab_temp, org_skl_temp, node_list_walked, seed_node, seed_node_dict, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info)
            tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]
            


        
        tree_new.relocation([-SHAPE[0],-SHAPE[1],-SHAPE[2]])
        for node in tree_new.get_node_list():
            node.set_z(round(node.get_z()/resize_radio,3))
            node.set_x(round(node.get_x()/resize_radio,3))
            node.set_y(round(node.get_y()/resize_radio,3))
            node.set_r(round(node.radius()/resize_radio,3))

        test_dir = predict_swc_path + image_name + '.pre_vector.v1.swc' # for road
        swc_save(tree_new, test_dir)

        end_time_a = time.time()






if __name__=='__main__':
    #######################################################
    #              load the config of model
    #######################################################
    args = config_3d.args

    #######################################################
    #              Checking if GPU is used
    #######################################################
    device_ids = [0, 1]

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')
    device = torch.device("cuda:" + str(args.gpu_id) if train_on_gpu else "cpu")

    
    #######################################################
    #              Setting up the model
    #######################################################
    model_Inputs = [PE_Net_3D]

    model_name = model_Inputs[0]



    #######################################################

    print('================================================')
    print('  status   = ' + str(args.train_or_test))
    print('  gpu_id   = ' + str(args.gpu_id))
    print(' img_path  = ' + str(args.dataset_img_path))
    print(' img_t_path= ' + str(args.dataset_img_test_path))
    print('model_name = ' + str(model_name))
    print('to_restore = ' + str(args.to_restore))
    print('batch_size = ' + str(args.batch_size))
    print('   epoch   = ' + str(args.epochs))
    print('    bins   = ' + str(args.vector_bins))
    print('================================================')
    
    train_or_test = str(args.train_or_test)

    if train_or_test == 'train':
        train(args, model_name, device_ids, device)
        print('train')
    elif train_or_test == 'test':
        test(args, model_name, device_ids, device)
        print('test')
    else:
        print("end")


# python train_3D.py --gpu_id 0 --train_or_test train
# python train_3D.py --gpu_id 0 --train_or_test test


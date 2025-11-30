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


from models.RPNet_2D import PE_Net_2D
from tools.tracing.tracing_tools_2D import get_pos_image_2d, get_network_predict_vecroad_2d, tracing_strategy_single_vecroad_2d_test
from tools.tracing.tracing_tools_2D import remove_isolated_noise, remove_noise
from tools.Data_Loader_2d import Images_Dataset_folder_2d
from tools.Losses import dice_loss, MSE_loss, L1_loss, dice_score, bce_loss_w

from skimage import morphology, transform

import time

from configs import config_2d
from lib.klib.baseio import *
from lib.swclib.swc_io import swc_save_metric, swc_save, read_swc_tree
from lib.swclib.swc_tree import SwcTree
from lib.swclib.swc_node import SwcNode
from lib.swclib.euclidean_point import EuclideanPoint
from lib.swclib.edge_match_utils import get_bounds
from lib.swclib import edge_match_utils
from lib.swclib.re_sample import up_sample_swc_tree

import copy


args = config_2d.args
resize_radio = args.resize_radio



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

    Training_Data = Images_Dataset_folder_2d(train_data_dir, train_label_dir, train_data_dir_list, train_label_dir_list, datasets_plag)

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


    Training_Data = Images_Dataset_folder_2d(train_data_dir, train_label_dir, train_data_dir_list, train_label_dir_list, mode = 'test')

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
    def model_unet_LSTM(model_input, in_channel, out_channel, freeze_net):
        model = model_input()
        return model

    model_train = model_unet_LSTM(model_name, 3, 1, freeze_plag)
    model_train = torch.nn.DataParallel(model_train, device_ids=device_ids)
    model_train.to(device)

    softmax = torch.nn.Softmax()#.cuda(gpu)
    criterion = torch.nn.CrossEntropyLoss()#.cuda(gpu)
    L1loss = torch.nn.L1Loss()
    L2loss = torch.nn.MSELoss()#.cuda(gpu)
    bce_criterion = torch.nn.BCELoss()
    sigmoid_layer = torch.nn.Sigmoid()

    
    NLL_criterion = torch.nn.NLLLoss()


    opt = torch.optim.Adam(model_train.parameters(), lr=initial_lr) # try SGD
    MAX_STEP = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-7)
    
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


            
            y_exist_skl = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            y_edge_topo = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            
            y_exist_point = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            y_exist_point_pred = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            
            y_exist_edge = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            y_exist_edge_pred = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            
            
            
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
            
            
            lossT =  loss_img_centerline  + loss_img_point + loss_img_edge + loss_img_edge_topo
            
            
        
            train_loss += lossT.item()
            lossT.backward()
            opt.step()


            train_end_time = time.time()
            if (train_step_temp+1) % 50 == 0:

                lossdice_centerline = dice_score(y_skl_pred, y_skl)
                lossdice_point = dice_score(y_exist_point_pred, y_exist_point)
                lossdice_edge = dice_score(y_exist_edge_pred, y_exist_edge)
                
                print('Epoch: {}/{} \t Step: {}/{} \t Total Loss: {:.5f} \t Time: {:.5f}/step'.format(i + 1, epoch, train_step_temp, total_step, lossT.item(),  train_end_time-train_begin_time))
                print('Centerline Loss: {:.5f}  F1: {:.5f}  | Point Loss: {:.5f}  F1: {:.5f} | Edge Loss: {:.5f}  F1: {:.5f} | Edge Topo Loss: {:.5f}'.format(loss_img_centerline.item(), lossdice_centerline.item(), loss_img_point.item(), lossdice_point.item(), loss_img_edge.item(), lossdice_edge.item(), loss_img_edge_topo.item()))

            train_step_temp += 1
            writer.add_scalar('Training Point Loss', loss_img_point.item(), global_step=global_step_)
            writer.add_scalar('Training Edge Loss', loss_img_edge.item(), global_step=global_step_)
            writer.add_scalar('Training Centerline Loss', loss_img_centerline.item(), global_step=global_step_)
            writer.add_scalar('Training Edge TOPO Dis Loss', loss_img_edge_topo.item(), global_step=global_step_)
            writer.add_scalar('Training Total Loss', lossT.item(), global_step=global_step_)
            global_step_ += 1

            load_begin_time_1 = time.time()
            # pause

        #######################################################
        #Validation Step
        #######################################################

        model_train.eval()
        torch.no_grad() #to increase the validation process uses less memory

        neg_num = 0
        neg_loss = 0

        pos_num = 0
        pos_loss = 0
        

        point_acc_list = []
        centerline_dis_acc_list = []
        centerline_acc_list = []
        edge_acc_list = []

        for x_img, y_skl, y_skl_b, x_walk, y_point, y_edge, x_exist in test_loader:
            train_begin_time = time.time()            
            exist_pos_id = np.where(x_exist==1)

            x_img, y_skl, y_skl_b, x_walk, y_point, y_edge = x_img.to(device), y_skl.to(device), y_skl_b.to(device), x_walk.to(device), y_point.to(device), y_edge.to(device)
            time.sleep(0.01) # 缓解GPU压力
            opt.zero_grad()
            
            y_skl_pred, y_point_pred, y_edge_pred = model_train(x_img, x_walk)
            
            y_exist_skl = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            y_exist_skl_lab = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
             
            
            y_exist_point = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            y_exist_point_pred = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            
            y_exist_edge = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            y_exist_edge_pred = torch.zeros([len(exist_pos_id[0]), 1, data_shape[0], data_shape[1]], dtype=torch.float32)
            
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
            loss_img_edge_topo = lambda_edge_topo * L1loss(y_exist_edge_pred - y_exist_skl* y_exist_edge_pred, y_exist_skl_lab)
            
            lossL =  loss_img_point + loss_img_centerline + loss_img_edge + loss_img_edge_topo
            

            valid_loss += lossL.item()

            valid_step_temp += 1

            # =========================================================================================

            # seg centerline
            lossdice_centerline = dice_score(y_skl_pred, y_skl)
            lossdice_point = dice_score(y_exist_point_pred, y_exist_point)
            lossdice_edge = dice_score(y_exist_edge_pred, y_exist_edge)
            
            centerline_acc_list.append(lossdice_centerline.item())
            point_acc_list.append(lossdice_point.item())
            edge_acc_list.append(lossdice_edge.item())

            
        print("============================================")
        print('total num: %d' % (split))
        print("--------------------------------------------")
        print('centerline acc:', np.mean(centerline_acc_list))
        print('node acc:', np.mean(point_acc_list))
        print('edge acc:', np.mean(edge_acc_list))
        print("============================================")

        writer.add_scalar('Test Loss', lossL.item(), global_step=i)


        #######################################################
        #               To write in Tensorboard
        #######################################################

        train_loss = train_loss / train_step_temp
        valid_loss = valid_loss / valid_step_temp
        valid_class_loss = valid_class_loss / valid_step_temp
        valid_reg_loss = valid_reg_loss / valid_step_temp

        if (i+1) % 1 == 0:
            print('Epoch: {}/{} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(i + 1, epoch, train_loss, valid_loss))
        #######################################################
        #                    Early Stopping
        #######################################################
        

        r_loss = 3 - np.mean(centerline_acc_list) - np.mean(point_acc_list) - np.mean(edge_acc_list)

        torch.save(model_train.state_dict(), MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '.pth')

        if r_loss <= r_loss_min: 
            print('R loss decreased (%6f --> %6f).  Saving model ' % (r_loss_min, r_loss))
            torch.save(model_train.state_dict(), MODEL_DIR  + str(epoch) + '_' + str(batch_size) + '/epoch_' + str(epoch) + '_batchsize_' + str(batch_size) + '_best.pth')
            r_loss_min = r_loss

        time_elapsed = time.time() - since
        print('this epoch time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        n_iter += 1
        


def test(args, model_name, device_ids, device):
    predict_swc_path = args.predict_swc_path
    if not os.path.exists(predict_swc_path):
        os.makedirs(predict_swc_path)
    
    predict_seed_path = args.predict_seed_path
    predict_centerline_path = args.predict_centerline_path

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

    stride_height = args.stride_height
    stride_width = args.stride_width

    node_r = args.r # Drive
    
    SHAPE = [data_shape[0],data_shape[1]]

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

        print(test_img_dir)
        img_rgb = open_tif(test_img_dir).astype(np.float32)
        stack_img = transform.resize(img_rgb,(round(img_rgb.shape[0]*resize_radio), round(img_rgb.shape[1]*resize_radio),3))


        seed_list_temp = [[],[]]
        stack_seed_swc_dir = predict_seed_path + image_name + '.swc' # for drive


        seed_tree = read_swc_tree(stack_seed_swc_dir)

        for tn in seed_tree.get_node_list():
            if tn.is_virtual():
                continue
            seed_list_temp[0].append(round(tn.get_y()*resize_radio)+SHAPE[0])
            seed_list_temp[1].append(round(tn.get_x()*resize_radio)+SHAPE[1])
        
        indices = list(range(len(seed_list_temp[0])))

        seed_list = [[],[]]
        for i in range(len(seed_list_temp[0])):
            num = indices[i]
            seed_list[0].append(seed_list_temp[0][num])
            seed_list[1].append(seed_list_temp[1][num])
            
            

        stack_skl_dir = predict_centerline_path + image_name + '.pro.skl.tif' # for road

        
        stack_skl = open_tif(stack_skl_dir).astype(np.float32)
        
        stack_skl_temp = copy.deepcopy(stack_skl)
        th = 128
        stack_skl_temp[stack_skl<=th] = 0 
        stack_skl_temp[stack_skl>th] = 255
        stack_skl = stack_skl_temp//255
        stack_skl = morphology.skeletonize(stack_skl)*1
        stack_skl_temp = copy.deepcopy(stack_skl)
        
        
        stack_lab = open_tif(stack_skl_dir).astype(np.float32)
        

        tracing_strategy_flag = 'netracer'

        
        # ===============resize the image=====================
        h, w, _ = stack_img.shape
        h_new = ((h-(test_patch_height-stride_height))//(stride_height)+1)*(stride_height) + (test_patch_height-stride_height)
        w_new = ((w-(test_patch_width-stride_width))//(stride_width)+1)*(stride_width) + (test_patch_width-stride_width)
        
        stack_img_new = np.zeros([h_new, w_new, 3],dtype=np.float32)
        stack_lab_new = np.zeros([h_new, w_new],dtype=np.uint8)
        stack_skl_new = np.zeros([h_new, w_new],dtype=np.uint8)

        stack_img_new[0:h,0:w,:] = copy.deepcopy(stack_img)
        stack_lab_new[0:h,0:w] = copy.deepcopy(stack_lab)
        stack_skl_new[0:h,0:w] = copy.deepcopy(stack_skl_temp)

        org_img_shape = stack_img_new.shape
        org_img_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1], 3])
        org_lab_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1]])
        org_skl_temp = np.zeros([org_img_shape[0]+2*SHAPE[0],org_img_shape[1]+2*SHAPE[1]])

        
        org_img_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1], :] = copy.deepcopy(stack_img_new)
        org_lab_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1]] = copy.deepcopy(stack_lab_new)
        org_skl_temp[1*SHAPE[0]:1*SHAPE[0] + org_img_shape[0], 1*SHAPE[1]:1*SHAPE[1] + org_img_shape[1]] = copy.deepcopy(stack_skl_new)


        shape_image = np.zeros([h_new, w_new],dtype=np.float32)
        # =====================================================
        
        seed_list_flag = np.ones(len(seed_list[0]))
        print('seed node num %d ' % (seed_list_flag.shape[0]))


        pos_list = []

        test_image = copy.deepcopy(org_img_temp).astype(np.uint8)

        # build tree
        tree_new = SwcTree()
        tree_new_covered = SwcTree()
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
            if i == 0:
                glob_node_id = 1
            else:
                try:
                    glob_node_id = max(tree_new.id_set) + 1
                except:
                    glob_node_id = 1
            
            print("------------------- tracing ----------------------", i + 1, " / ", seed_list_flag.shape[0])

            seed_node_z = 0
            seed_node_x = seed_list[0][i]
            seed_node_y = seed_list[1][i]

            
            
            # init seed node
            seed_node_dict = {'node_id': glob_node_id, 'node_z': seed_node_z, 'node_x': seed_node_x, 'node_y': seed_node_y, 'node_r': node_r, 'node_p_id': -1, 'parent_node': tree_new._root, 'node_exist': 255, 'node_list_walked':[[seed_node_z, seed_node_x, seed_node_y]]}
            seed_node_dict['direction_vector'] = [0,0,0]

            seed_node = SwcNode(nid=seed_node_dict['node_id'], ntype=0, center=EuclideanPoint(center=[seed_node_dict['node_y'],seed_node_dict['node_x'],seed_node_dict['node_z']]), radius=round(seed_node_dict['node_r'],3))
            
            node_temp_list = edge_match_utils.get_nearby_edges(rtree=tree_new_rtree, point=seed_node, id_edge_dict=tree_new_idedge_dict, threshold=node_r)
            
            if len(node_temp_list) != 0:
                if node_temp_list[0][1]<= node_r*2:
                    print("this seed is already traced")
                    continue
                else:
                    print("can be trace 1")
            else:
                print("can be trace 2")
            
            
            node_list_walked = seed_node_dict['node_list_walked']
            seed_node_img, seed_node_img_walked = get_pos_image_2d(org_img_temp, node_list_walked, [seed_node_z, seed_node_x, seed_node_y], SHAPE)
            seed_node_img = np.transpose(seed_node_img, (2, 0, 1))
            seed_node_img_walked = seed_node_img_walked.reshape(1,*SHAPE)
            
            exist, next_node_pos = get_network_predict_vecroad_2d(org_lab_temp, org_skl_temp, seed_node_img, node_list_walked, seed_node_img_walked, SHAPE, model_test, device, vector_bins, tracing_strategy_flag)
            
            
            seed_node.parent = seed_node_dict['parent_node']
            
            # start tracing
            tree_new, r_tree_info = tracing_strategy_single_vecroad_2d_test(org_img_temp, org_lab_temp, org_skl_temp, node_list_walked, seed_node, seed_node_dict, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info)
            tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]
           

            

            
        
        tree_new.relocation([0,-SHAPE[0],-SHAPE[1]])
        for node in tree_new.get_node_list():
            node.set_z(round(node.get_z()/resize_radio,3))
            node.set_x(round(node.get_x()/resize_radio,3))
            node.set_y(round(node.get_y()/resize_radio,3))
            node.set_r(round(node.radius()/resize_radio,3))
        
        test_dir = predict_swc_path + image_name + '.pre_vector.road.v1.'+tracing_strategy_flag+'.swc' 
        swc_save(tree_new, test_dir)
        
        

        end_time_a = time.time()
        print('time:', end_time_a-begin_time_a)




if __name__=='__main__':
    #######################################################
    #              load the config of model
    #######################################################
    args = config_2d.args

    #######################################################
    #              Checking if GPU is used
    #######################################################
    device_ids = [2,3]
    cuda_id = 2

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:" + str(cuda_id) if train_on_gpu else "cpu")
    torch.cuda.set_device(cuda_id)
    
    #######################################################
    #              Setting up the model
    #######################################################
    model_Inputs = [PE_Net_2D]

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



# python train_2D.py --gpu_id 0 --train_or_test train
# python train_2D.py --gpu_id 0 --train_or_test test

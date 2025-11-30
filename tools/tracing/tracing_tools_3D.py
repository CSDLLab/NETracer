import torchvision
import torch
import torch.nn.functional as F

from scipy import ndimage
from scipy.spatial import distance_matrix
from scipy.ndimage import morphology, generate_binary_structure
from skimage import morphology
import numpy as np
from numpy import linalg as LA
import copy
from rtree import index
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tifffile

from lib.klib.glib.SWCExtractor import Vertex
from lib.klib.glib.Obj3D import Point3D, Sphere, Cone
from lib.klib.baseio import *
from lib.swclib import edge_match_utils
from lib.swclib.swc_node import SwcNode
from lib.swclib.euclidean_point import EuclideanPoint

from lib.swclib.swc_io import swc_save

import time

from configs import config_3d

import sys
sys.setrecursionlimit(1000000)


args = config_3d.args
resize_radio = args.resize_radio
r_resize = args.r_resize
data_shape = args.data_shape



def gen_circle_gaussian_3d(size=[data_shape[0],data_shape[1],data_shape[2]], r=32.0, z_offset=0, x_offset=0, y_offset=0):
    z0 = size[0]//2
    x0 = size[1]//2
    y0 = size[2]//2
    x0 += x_offset
    y0 += y_offset
    z0 += z_offset
    z, x, y = np.ogrid[:size[0], :size[1], :size[2]]
    image = 1 * np.exp(
        -(((x - x0) ** 2 / (2 * r ** 2)) + ((y - y0) ** 2 / (2 * r ** 2)) + ((z - z0) ** 2 / (2 * r ** 2))))
    return image.astype(np.float32)




def find_if_parent(node_A, node_B, k = 3):
	# find if nodeA is nodeB's parent, k is num
	node_current = node_B
	node_target = node_A

	for i in range(k):
		# print(node_current, node_current.parent)
		if node_current.parent == node_target:
			return True
		else:
			# print(node_current, node_current.parent)
			node_current = node_current.parent
			
			# if node_current.get_id() == 1:
			# 	return True
			if node_current.is_virtual():# 找到了根节点
				return False
	return False

def find_close_point(node_current, node_a, node_b):
	node_current_pos = [node_current.get_z(), node_current.get_x(),node_current.get_y()]
	node_a_pos = [node_a.get_z(), node_a.get_x(), node_a.get_y()]
	node_b_pos = [node_b.get_z(), node_b.get_x(), node_b.get_y()]

	distance_a = math.sqrt((node_current_pos[0] - node_a_pos[0]) ** 2 + (node_current_pos[1] - node_a_pos[1]) ** 2 + (
				node_current_pos[2] - node_a_pos[2])**2)
	distance_b = math.sqrt((node_current_pos[0] - node_b_pos[0]) ** 2 + (node_current_pos[1] - node_b_pos[1]) ** 2 + (
				node_current_pos[2] - node_b_pos[2]) ** 2)
	if distance_a < distance_b:
		return node_a
	else:
		return node_b

def o_distance(point_A, point_B):
	distance = math.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2 + (point_A[2] - point_B[2]) ** 2)
	return distance

def in_range(n, start, end = 0):
    return start <= n <= end if end >= start else end <= n <= start

def cosine_similarity(vector_A,vector_B):
    numerator = 0
    denominator_A = 0
    denominator_B = 0
    for i in range(len(vector_A)):
        numerator = numerator + vector_A[i]*vector_B[i]
        denominator_A = denominator_A + vector_A[i] ** 2
        denominator_B = denominator_B + vector_B[i] ** 2
    result = numerator/np.sqrt(denominator_A+1e-6)/np.sqrt(denominator_B+1e-6)
    return result

def binary_image_skeletonize(input):
    skl = np.expand_dims(input,0)
    label_temp_dilation = morphology.binary_dilation(skl, morphology.ball(2))
    label_temp_erosion = morphology.binary_erosion(label_temp_dilation, morphology.ball(2))

    skeletonize = morphology.skeletonize_3d(label_temp_erosion * 1) * 255
    return skeletonize

def get_node_centerline_from_image_3d(node_pos, r_temp, org_image_centerline):
    z_temp = int(round(node_pos[0], 0))
    x_temp = int(round(node_pos[1], 0))
    y_temp = int(round(node_pos[2], 0))

    # centerline 扩大范围
    lambda_ = 1.0
    r_temp_ = int(round((r_temp) * lambda_, 0))


    radius_region = np.zeros([2 * r_temp_ + 1, 2 * r_temp_ + 1, 2 * r_temp_ + 1])
    if z_temp - r_temp_ >= 0 and z_temp + r_temp_ < org_image_centerline.shape[0] and x_temp - r_temp_ >= 0 and x_temp + r_temp_ < org_image_centerline.shape[1] and y_temp - r_temp_ >= 0 and y_temp + r_temp_ < org_image_centerline.shape[2]:
        centerline_temp = copy.deepcopy(org_image_centerline[z_temp - r_temp_:z_temp + r_temp_ + 1, x_temp - r_temp_:x_temp + r_temp_ + 1, y_temp - r_temp_:y_temp + r_temp_ + 1])
        # print(np.sum(centerline_temp))
        centerline_temp_point_list = np.where(centerline_temp == 1)
    else:
        centerline_temp = np.zeros_like(radius_region)
        # centerline_temp[r_temp_][r_temp_][r_temp_] = 0#org_image_centerline[z_temp][x_temp][y_temp]
        centerline_temp_point_list = np.where(centerline_temp == 1)

    # 矫正归位
    for i in range(len(centerline_temp_point_list[0])):
        centerline_temp_point_list[0][i] = centerline_temp_point_list[0][i] - r_temp_
        centerline_temp_point_list[1][i] = centerline_temp_point_list[1][i] - r_temp_
        centerline_temp_point_list[2][i] = centerline_temp_point_list[2][i] - r_temp_

    return centerline_temp_point_list

def get_bounds(point_a, point_b, extra=0):
    """
    get bounding box of a segment
    Args:
        point_a: two points to identify the square
        point_b:
        extra: float, a threshold
    Return:
        res(tuple):
    """
    point_a = np.array(point_a.get_center()._pos)
    point_b = np.array(point_b.get_center()._pos)
    res = (np.where(point_a > point_b, point_b, point_a) - extra).tolist() + (np.where(point_a > point_b, point_a, point_b) + extra).tolist()

    return tuple(res)


structure = generate_binary_structure(3, 1)

def find_path_edge(img, seed, end_point_list):
    # begin_time_a = time.time()
    
    img = img.astype(np.float32)
    img_max = np.max(img)
    img_min = np.min(img)
    range_ = img_max - img_min

    score_img = np.full_like(img, np.inf)
    score_img[seed[0]][seed[1]][seed[2]] = 1

    path_img = np.ones_like(score_img).astype(np.float32)
    path_img[seed[0]][seed[1]][seed[2]] = np.inf

    walked_img = np.full((img.shape[0] + 2, img.shape[1] + 2, img.shape[2] + 2), np.inf, dtype=np.float32)
    walked_img[1:-1,1:-1,1:-1] = 1
    walked_img[seed[0]+1][seed[1]+1][seed[2]+1] = np.inf


    begin_time_a = time.time()
    seq_img = np.zeros_like(score_img).astype(np.float32)
    seq_len_img = np.zeros_like(score_img).astype(np.float32)

    begin_time_b = time.time()
    print('stage 0-1', begin_time_b - begin_time_a)

    stage_1_1 = 0
    stage_1_2 = 0
    stage_1_3 = 0
    stage_1_3_ = 0
    
    max_value_old = np.inf
    z_new_temp_old = seed[0]
    x_new_temp_old = seed[1]
    y_new_temp_old = seed[2]
    
    k = 0
    # id = 1
    next_node_pos = seed

    while k < 10000:
        z_temp = next_node_pos[0]
        x_temp = next_node_pos[1]
        y_temp = next_node_pos[2]



        

        begin_time_a = time.time()
        structure_path_temp = copy.deepcopy(walked_img[z_temp - 1 + 1:z_temp + 2 + 1, x_temp - 1 + 1:x_temp + 2 + 1,y_temp - 1 + 1:y_temp + 2 + 1]) * structure
        next_node_list_temp = np.where(structure_path_temp == 1)
 
        begin_time_b = time.time()
        stage_1_1 += begin_time_b - begin_time_a
        
        
        
        
        
        begin_time_a = time.time()
        for node_temp_id in range(len(next_node_list_temp[0])):
            z_ = next_node_list_temp[0][node_temp_id] + z_temp - 1
            x_ = next_node_list_temp[1][node_temp_id] + x_temp - 1
            y_ = next_node_list_temp[2][node_temp_id] + y_temp - 1

            score_current = score_img[z_][x_][y_]

            
            
            k_ = 10
            score_1 = np.exp(k_*((img[z_][x_][y_] - img_min) / range_) ) -1
            score_2 = np.exp(k_*((img[z_temp][x_temp][y_temp] - img_min) / range_) ) -1
            score_new = score_img[z_temp][x_temp][y_temp] + (score_1 + score_2)/2


            score_img[z_][x_][y_] = min(score_current, score_new)

            walked_img[z_ + 1][x_ + 1][y_ + 1] = np.inf
        
        begin_time_b = time.time()
        stage_1_2 += begin_time_b - begin_time_a
        
        
        
       
        #更新当前位置，已经遍历
        walked_img[z_temp+1][x_temp+1][y_temp+1] = np.inf
        
        
        
        max_index = np.argmin(score_img*path_img)

        z_new, x_new, y_new = np.unravel_index(max_index, score_img.shape)
        path_img[z_new][x_new][y_new] = np.inf
        



        next_node_pos = [z_new, x_new, y_new]

        if next_node_pos in end_point_list:
            path_img[z_new][x_new][y_new] = np.inf
            break

        if score_img[z_new][x_new][y_new]==255:
            break


        path_img[z_temp][x_temp][y_temp] = np.inf

        k=k+1   



    return path_img, score_img, seq_img, seq_len_img, next_node_pos
     


def tracing_strategy(current_node_dict, tree_new_rtree, tree_new_idedge_dict):
    node_id = current_node_dict['node_id']
    z_temp_new_ave = current_node_dict['node_z']
    x_temp_new_ave = current_node_dict['node_x']
    y_temp_new_ave = current_node_dict['node_y']
    node_r = 2.0 * 1.5
    
    parent_node = current_node_dict['parent_node']
    node_exist = current_node_dict['node_exist']
    
    if node_id == 1: # 第一步忽略
        node_range = node_r
        distance_to_nearby_branch=np.Inf
        
    elif node_id == 2: # 第二步（由于无法edge_match_utils.get_nearby_edges计算距离，因此单独设置）
        node_new = EuclideanPoint(center=[y_temp_new_ave,x_temp_new_ave,z_temp_new_ave])
        node_parent = EuclideanPoint(center=[parent_node.get_x(),parent_node.get_y(),parent_node.get_z()])
        distance_to_nearby_branch = node_new.distance_to_point(node_parent)
        node_range = node_r
    else:
        son_node_temp = SwcNode(nid=1, ntype=0, center=EuclideanPoint(center=[y_temp_new_ave,x_temp_new_ave,z_temp_new_ave]), radius=node_r)
        node_temp_list = edge_match_utils.get_nearby_edges(rtree=tree_new_rtree, point=son_node_temp, id_edge_dict=tree_new_idedge_dict, threshold=node_r*3)#三倍半径内的枝干
         
        # print(len(node_temp_list))
        
        if len(node_temp_list) == 0:
            node_range = node_r
            # distance_to_nearby_branch = -1 # 可能产生了大跳跃，因此停止
            distance_to_nearby_branch = np.Inf # 可能产生了大跳跃
        else:

            distance_to_nearby_branch = node_temp_list[0][1]
            
            
            node_range = node_r
            
            if son_node_temp._pos.get_x() == parent_node._pos.get_x() and son_node_temp._pos.get_y() == parent_node._pos.get_y() and son_node_temp._pos.get_z() == parent_node._pos.get_z():
                node_range = np.Inf
            

    # 中止判断条件
    boundary_th = 128 # should be adjust
    if node_exist < boundary_th:
        print("存在越界", node_exist)
        end_tracing = True
        end_tracing_code = '01'
        return end_tracing, end_tracing_code
    elif distance_to_nearby_branch < node_range:
        if distance_to_nearby_branch == -1:
            print("this may be a large gap")
            end_tracing_code = '02'
            # end_tracing = True
        else:
            print("this position is traced")
            end_tracing_code = '03'
            # end_tracing = False
        end_tracing = True
        return end_tracing, end_tracing_code
    else:
        end_tracing_code = '00'
        end_tracing = False
        return end_tracing, end_tracing_code
        

        
# 根据网络架构调整
def get_pos_image_3d(image, node_list, pos, shape):
    z_half = shape[0]//2
    x_half = shape[1]//2
    y_half = shape[2]//2
    pos_z, pos_x, pos_y = pos

    node_img = image[pos_z- z_half:pos_z + z_half, pos_x- x_half:pos_x + x_half, pos_y-y_half:pos_y + y_half].copy()

    mark = np.zeros((data_shape[0],data_shape[1],data_shape[2]))
    mark_shape = (data_shape[0],data_shape[1],data_shape[2])
    
    start_point = [z_half, x_half, y_half]  # (x, y) 起点坐标
    start_node = Vertex(0, 0, start_point[0], start_point[1], start_point[2], 1, -1)
    if len(node_list)<2:
        setMarkWithSphere(mark, Sphere(Point3D(*start_node.pos), start_node.r), mark_shape)
    else:
        for i in range(2, len(node_list)+1):
            end_point = node_list[-i]
            
            end_node = Vertex(0, 0, end_point[0]-node_list[-1][0]+data_shape[0]//2, end_point[1]-node_list[-1][1]+data_shape[1]//2, end_point[2]-node_list[-1][2]+data_shape[2]//2, 1, -1)

            setMarkWithCone(mark, Cone(Point3D(*start_node.pos), 1, Point3D(*end_node.pos), 1), mark_shape)
            start_node = end_node
            
    img_walk = np.array(mark).astype(np.uint8)

    
    return node_img, img_walk

def get_network_predict_vecroad_3d(org_lab_temp, org_skl_temp, image, node_list_walked, image_walk, shape, model_test, device, vector_bins, tracing_strategy_flag, time_=None): 
    if time_ == None:
        predicted_time = 0
        search_time = 0
    else:
        predicted_time, search_time = time_[0], time_[1]
    
    image = np.sqrt(copy.deepcopy(image)) / 255 #* 2 - 1
    image_walk = copy.deepcopy(image_walk) / 255
    
    data_transform = torchvision.transforms.Compose([])
    
    # seq_len, _,_,_ = image.shape
    image_tensor = np.zeros(dtype=np.float32, shape=[1, 2, *shape])
    image_tensor[0,0,:,:,:] = copy.deepcopy(image[0])
    image_tensor[0,1,:,:,:] = copy.deepcopy(image_walk[0])
        
        
    softmax = torch.nn.Softmax()

    
    image_tensor_input = data_transform(image_tensor)
    test_loader = torch.utils.data.DataLoader(image_tensor_input, batch_size=1)

    
    model_test.eval()
    torch.no_grad() #to increase the validation process uses less memory

    for x_batch in test_loader:
        batch_input = x_batch.to(device)

        img_org, img_walk = batch_input[0][0], batch_input[0][1]
        
        img_org = img_org.reshape(1,1,data_shape[0],data_shape[1],data_shape[2])
        img_walk = img_walk.reshape(1,1,data_shape[0],data_shape[1],data_shape[2])
        

        pre_time_a = time.time()
        # 网络预测
        y_skl_pred, y_point_pred, y_edge_pred = model_test(img_org, img_walk) # v7-m2
        pre_time_b = time.time()
        predicted_time += pre_time_b-pre_time_a
        
        # Point预处理
        point_temp_mask = gen_circle_gaussian_3d(size = [16,64,64], r=2.0, z_offset=0, x_offset=0, y_offset=0)*255
        point_temp_mask[point_temp_mask<=128]=0
        point_temp_mask[point_temp_mask>128]=255
        point_temp_mask = point_temp_mask/255
                
        pred_tb = y_point_pred.cpu().detach().numpy()
        pred_tb = pred_tb.reshape(data_shape[0],data_shape[1],data_shape[2])
        mask = np.zeros_like(pred_tb)
        mask[1:-1, 4:-4, 4:-4] = 1
        pred_tb = mask*pred_tb*255
        
        pred_tb = (1 - point_temp_mask)*pred_tb
        
        # Edge预处理
        pred_edge = y_edge_pred.cpu().detach().numpy()*255
        pred_edge = pred_edge.reshape(data_shape[0],data_shape[1],data_shape[2]).astype(np.uint8)
        
        
        search_time_a = time.time()
        # 选取结点保存到next_node_pos
        next_node_pos_patch = []
        next_node_pos = []
        if np.sum(pred_tb) == 0:
            max_depth = 0
            max_row = 0
            max_col = 0
            next_node_pos.append([max_depth+node_list_walked[-1][0], max_row+node_list_walked[-1][1], max_col+node_list_walked[-1][2]])
        else:
            point_th = np.max(pred_tb)*0.5
            pred_tb_temp = copy.deepcopy(pred_tb)
            while np.max(pred_tb_temp)>point_th:
                max_index = np.argmax(pred_tb_temp)
                max_depth, max_row, max_col = np.unravel_index(max_index, pred_tb_temp.shape)
                next_node_pos_patch.append([max_depth, max_row, max_col])
                
                max_depth = max_depth - data_shape[0]//2
                max_row = max_row - data_shape[1]//2
                max_col = max_col - data_shape[2]//2

                
                point_temp_mask = gen_circle_gaussian_3d(size = [16,64,64], r=2.0, z_offset=max_depth, x_offset=max_row, y_offset=max_col)*255

                point_temp_mask[point_temp_mask<=50]=0
                point_temp_mask[point_temp_mask>50]=255
                point_temp_mask = point_temp_mask/255
                point_temp_mask = point_temp_mask.astype(np.uint8)
                pred_tb_temp = (1 - point_temp_mask)*pred_tb_temp
                

        search_time_b = time.time()
        search_time_a_b = search_time_b - search_time_a
        
        
    
        
        # 保存结果
        img_org = img_org.cpu().detach().numpy()
        img_org = img_org.reshape(data_shape[0],data_shape[1],data_shape[2])
        img_org = img_org*255 
        img_org = img_org.astype(np.uint16)
        
        img_walk = img_walk.cpu().detach().numpy()
        img_walk = img_walk.reshape(data_shape[0],data_shape[1],data_shape[2])
        img_walk = img_walk*255
        img_walk = img_walk.astype(np.uint8)
        
        
        


        path_img, score_img , _ ,_, next_node_pos_ = find_path_edge(pred_edge, seed, next_node_pos_patch)

        
        
        search_time_d = time.time()
        search_time_c_d = search_time_d - search_time_c
        
        search_time += search_time_c_d
        

        total_time = [predicted_time, search_time]
        
        
        
        

                
        max_depth = next_node_pos_[0] - data_shape[0]//2 
        max_row = next_node_pos_[1] - data_shape[1]//2 
        max_col = next_node_pos_[2] - data_shape[2]//2 
        
        next_node_pos_ = [0,0,0]
        next_node_pos_[0] = max_depth  + node_list_walked[-1][0]
        next_node_pos_[1] = max_row  + node_list_walked[-1][1]
        next_node_pos_[2] = max_col  + node_list_walked[-1][2]
        
        next_node_pos.append(next_node_pos_)
        
        next_node_pos_ = [0,0,0]
        next_node_pos_[0] = -max_depth  +node_list_walked[-1][0]
        next_node_pos_[1] = -max_row  +node_list_walked[-1][1]
        next_node_pos_[2] = -max_col  +node_list_walked[-1][2]
        
        next_node_pos.append(next_node_pos_)
        
        
        next_node_pos_final = []
        next_node_exist_final = []
        
        for node_pos_temp in next_node_pos:
            z_temp_new = node_pos_temp[0]
            x_temp_new = node_pos_temp[1]
            y_temp_new = node_pos_temp[2]

            node_pos_next = [round(z_temp_new), round(x_temp_new), round(y_temp_new)]

            
            z_temp_new_ave = z_temp_new
            x_temp_new_ave = x_temp_new
            y_temp_new_ave = y_temp_new

    
            node_pos_next = [round(z_temp_new_ave), round(x_temp_new_ave), round(y_temp_new_ave)]
            node_exist_next = np.max(org_lab_temp[node_pos_next[0]-2:node_pos_next[0]+2, node_pos_next[1]-2:node_pos_next[1]+2,node_pos_next[2]-2:node_pos_next[2]+2])
            next_node_pos_final.append(node_pos_next)   
            next_node_exist_final.append(node_exist_next)   
        if time_ == None:
            return next_node_exist_final, next_node_pos_final
        else:
            return next_node_exist_final, next_node_pos_final, total_time


def tracing_strategy_single_vecroad_3d_test(org_image, org_lab_temp, org_skl_temp, node_list_walked, seed_node, seed_node_dict, tree_new, tracing_strategy_flag, device_info, data_info, r_tree_info, total_time):
    model_test, device = device_info[0], device_info[1]
    SHAPE, vector_bins = data_info[0], data_info[1]
    tree_new_rtree, tree_new_idedge_dict = r_tree_info[0], r_tree_info[1]
    
    img_shape = org_image.shape

    branch_node_list = []
    exist_score_list = []
    end_tracing = False
    
    
    current_node_dict = copy.deepcopy(seed_node_dict)
    
    node_pool.append(current_node_dict)
    # traced_pool = []
    
    steps = 1
    begin_node_id = tree_new.size()+1
    

    
    
    # 开始追踪
    while len(node_pool) != 0:
        print(begin_node_id, '--------------------------------', steps)
        
        # 从node_pool中取一个节点
        current_node_dict = node_pool.pop()
        
        end_tracing, end_tracing_code = tracing_strategy(current_node_dict, tree_new_rtree, tree_new_idedge_dict)
        

        if end_tracing == True:
            
            print('当前点不满足条件，跳过，换下一个点')
            continue
        else:
            # 当前节点信息
            # node_id = current_node_dict['node_id']
            node_id = begin_node_id
            node_parent_id = current_node_dict['node_p_id']
        
            current_z_temp = round(current_node_dict['node_z'])
            current_x_temp = round(current_node_dict['node_x'])
            current_y_temp = round(current_node_dict['node_y'])
        
            
            
            node_r = current_node_dict['node_r']
            node_list_walked = current_node_dict['node_list_walked']
            current_vector = current_node_dict['direction_vector']
            parent_node = current_node_dict['parent_node']
            
            # 打印信息
            # print('--------------------')
            # print('输出当前节点的信息：')
            # print('编号：', node_id)
            # print('位置：', current_z_temp-16, current_x_temp-64, current_y_temp-64)
            # print('编号(p)：', node_parent_id)
            # print('p：', parent_node)
            # print('walked：', node_list_walked)
            # print('--------------------')

            # 将当前结点加入追踪结果中
            if steps == 1:
                son_node = seed_node
                tree_new.id_set.add(seed_node_dict['node_id'])
                tree_new.get_node_list(update=True)
            else:
                son_node = SwcNode(nid=begin_node_id, ntype=0, center=EuclideanPoint(center=[round(current_y_temp,3),round(current_x_temp,3),round(current_z_temp,3)]), radius=round(node_r,3))
                tree_new.add_child(parent_node, son_node)
                tree_new.get_node_list(update=True)
                tree_new_rtree.insert(son_node.get_id(), get_bounds(son_node, son_node.parent, extra=node_r*1.5))
                tree_new_idedge_dict[son_node.get_id()] = tuple([son_node, son_node.parent])

            steps = steps + 1
            begin_node_id = begin_node_id+1
            

        
            # 未来点的预测
            node_pos = [current_z_temp, current_x_temp, current_y_temp]    
            seed_node_img, seed_node_img_walked = get_pos_image_3d(org_image, node_list_walked, node_pos, SHAPE)
            seed_node_img = seed_node_img.reshape(1,*SHAPE)
            seed_node_img_walked = seed_node_img_walked.reshape(1,*SHAPE)
            exist, next_node_pos, total_time = get_network_predict_vecroad_3d(org_lab_temp, org_skl_temp, seed_node_img, node_list_walked, seed_node_img_walked, SHAPE, model_test, device, vector_bins, tracing_strategy_flag, total_time)
    
            # 按照方向相似度,对结点重排序
            cos_sim_list = []
            node_pos_temp_id=0
            for node_pos_temp in next_node_pos:
                z_temp_new = node_pos_temp[0]
                x_temp_new = node_pos_temp[1]
                y_temp_new = node_pos_temp[2]
                next_vector = [z_temp_new-current_z_temp, x_temp_new-current_x_temp, y_temp_new-current_y_temp]
                
                cos_sim = cosine_similarity(current_vector, next_vector)
                cos_sim_list.append(cos_sim)
                # print('方向相似度：', cos_sim)
            
            def sort_and_return_index(lst):
                sorted_lst = sorted(lst)
                index_lst = [i[0] for i in sorted(enumerate(lst), key=lambda x:x[1])]
                return sorted_lst, index_lst

            cos_sim_sorted_list, cos_sim_sorted_list_index = sort_and_return_index(cos_sim_list)
            # print(cos_sim_sorted_list)
            # print(cos_sim_sorted_list_index)

            # 按照相似度低到高排序，依次添加========
            for node_temp_id in cos_sim_sorted_list_index:
                # 如果偏离完全相反，则跳过
                if cos_sim_list[node_temp_id] < 0: 
                    # print('方向相反，已跳过')
                    continue

                z_temp_new_ave = next_node_pos[node_temp_id][0]
                x_temp_new_ave = next_node_pos[node_temp_id][1]
                y_temp_new_ave = next_node_pos[node_temp_id][2]
                node_exist = exist[node_temp_id]
        
                
            
                node_pos_next = [round(z_temp_new_ave), round(x_temp_new_ave), round(y_temp_new_ave)]
                print('预测点位置：', node_temp_id+1,"|", round(z_temp_new_ave)-data_shape[0], round(x_temp_new_ave)-data_shape[1], round(y_temp_new_ave)-data_shape[2])

                # 判断是否在图像区域外
                if in_range(round(z_temp_new_ave), data_shape[0]//2, org_image.shape[0]-data_shape[0]//2) is False or in_range(round(x_temp_new_ave), data_shape[1]//2, org_image.shape[1]-data_shape[1]//2) is False or in_range(round(y_temp_new_ave), data_shape[2]//2, org_image.shape[2]-data_shape[2]//2) is False:
                    print("exceed the bound")
                    continue
                # 判断是否离父节点太远，要求10倍半径内
                dis_temp = o_distance(node_pos, node_pos_next)
                if dis_temp > 10*node_r:
                    print("too far from its parent!")
                    continue
                # 判断是否和父节点位置一样
                if round(z_temp_new_ave) == current_z_temp and round(x_temp_new_ave) == current_x_temp and round(y_temp_new_ave) == current_y_temp:
                    print("same as parent node!")
                    continue
                
                # 更新结点信息
                # 更新node_list_walked信息, 历史信息长度为10
                node_list_walked_len = 15
                node_list_walked_temp = copy.deepcopy(node_list_walked)
                if len(node_list_walked_temp)<node_list_walked_len:
                    node_list_walked_temp.append((z_temp_new_ave, x_temp_new_ave, y_temp_new_ave))
                else:
                    node_list_walked_temp.remove(node_list_walked_temp[0])
                    node_list_walked_temp.append((z_temp_new_ave, x_temp_new_ave, y_temp_new_ave))
                # print(node_list_walked)
                
                temp_node_dict = {'node_id': begin_node_id}
                
                temp_node_dict['node_id'] = begin_node_id
                temp_node_dict['node_z'] = z_temp_new_ave
                temp_node_dict['node_x'] = x_temp_new_ave
                temp_node_dict['node_y'] = y_temp_new_ave
                
                temp_node_dict['node_r'] = 2.0
                temp_node_dict['direction_vector'] = [z_temp_new_ave-current_z_temp, x_temp_new_ave-current_x_temp, y_temp_new_ave-current_y_temp]
                
                
                temp_node_dict['node_p_id'] = current_node_dict['node_id']
                
                temp_node_dict['parent_node'] = son_node
                temp_node_dict['node_exist'] = node_exist
                temp_node_dict['node_list_walked'] = node_list_walked_temp

                # print(temp_node_dict)
                
                node_pool.append(temp_node_dict)

            
        
    r_tree_info = [tree_new_rtree, tree_new_idedge_dict]
    return tree_new, r_tree_info, total_time
    

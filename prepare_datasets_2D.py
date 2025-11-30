import numpy as np
from lib.klib.baseio import *
from scipy.ndimage import filters as ndfilter
from scipy import ndimage
from lib.klib.glib.DrawSimulationSWCModel import simulate3DTreeModel_dendrite, save_swc

from lib.swclib.swc_io import swc_save, swc_save_preorder, read_swc_tree, read_swc_tree_matrix, swc_save_metric
from lib.swclib.swc_tree import SwcTree
from lib.swclib.swc_node import SwcNode
from lib.swclib.re_sample import up_sample_swc_tree, down_sample_swc_tree

from lib.swclib import euclidean_point
from lib.swclib import edge_match_utils, point_match_utils

import copy
import cv2 as cv
import multiprocessing as mp
from skimage import morphology, transform
import queue
import time
import math
import random
import tifffile
import GeodisTK
import os
from shutil import rmtree
from PIL import Image
import sys
sys.setrecursionlimit(100000)

import argparse

from PIL import Image, ImageDraw

# python prepare_datasets_2D.py --datasets_name ROAD --train_dataset_root_dir /8T1/xxx/datasets/ROAD/pointedgenet/training_data_v2/
# python prepare_datasets_2D.py --datasets_name DRIVE --train_dataset_root_dir /home/xxx/datasets/DRIVE/pointedgenet/training_data_v2/
# python prepare_datasets_2D.py --datasets_name DRIVE --train_dataset_root_dir /home/xxx/datasets/DRIVE/pointedgenet/training_data_v2/

def parse_args():
	parser = argparse.ArgumentParser()

	# (input dir) orginal data
	parser.add_argument('--datasets_name', default='ROAD',help='datasets name') # CHASEDB1

	parser.add_argument('--image_dir', default='/8T1/xxx/', help='orginal image saved here')

	# (output dir)
	parser.add_argument('--train_dataset_root_dir', default='/8T1/xxx/datasets/ROAD/pointedgenet/training_data/',help='orginal centerline saved here')
	parser.add_argument('--N_patches', default=150000,help='Number of training image patches') # 150000

	parser.add_argument('--input_dim', type=int, default=(64,64))
	parser.add_argument('--multi_cpu', type=int, default=20)

	args = parser.parse_args()
	
	return args


def convert_gif2tif(input_dir, resize_radio = 2):
	resize_radio_ = 2
	img_org = Image.open(input_dir)
	img = np.asarray(img_org)

	img_new = np.zeros([1, round(img.shape[0]*resize_radio), round(img.shape[1]*resize_radio)],dtype=np.uint8)
	img_new[0] = np.asarray(img_org.resize((round(img.shape[1]*resize_radio), round(img.shape[0]*resize_radio))))


	img_resize = img_org.resize((img.shape[1]*resize_radio_, img.shape[0]*resize_radio_))
	img_resize = np.asarray(img_resize)

	img_resize_new = np.zeros([1, img.shape[0]*resize_radio_, img.shape[1]*resize_radio_],dtype=np.uint8)
	img_resize_new[0] = img_resize
	img_resize_new[img_resize_new>=128] = 1
	img_resize_new[img_resize_new<128] = 0
	
	return img_new, img_resize_new

def convert_gif2skl(input_dir, output_dir):
	img = np.asarray(Image.open(input_dir))//255
	img_skl = morphology.skeletonize(img).astype(np.uint8)

	kernel = morphology.disk(2)
	img_skl2 = morphology.dilation(img_skl,kernel)
	img_skl2 = morphology.skeletonize(img_skl2).astype(np.uint8)
	img_skl2 = morphology.dilation(img_skl2,kernel)
	img_skl2 = morphology.skeletonize(img_skl2).astype(np.uint8)
	tifffile.imsave(output_dir, img_skl2)

	return 0 

def convert_skl2swc(input_dir, output_dir):
	img_skl = open_tif(input_dir)
	centerline_temp = np.where(img_skl==1)

	centerline = np.zeros([len(centerline_temp[0]),2],dtype=np.int)

	centerline[:, 0] = np.array(centerline_temp[0])
	centerline[:, 1] = np.array(centerline_temp[1])
	centerline_sample_branch, centerline_sample_leaf, branch_list_index, leaf_list_index = reconstruction_tif2swc(img_skl, centerline)


	return 0 

def dataset_normalized(imgs):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized

def adjust_gamma(imgs, gamma=1.0):
	# assert (len(imgs.shape)==4)  #4D arrays
	# assert (imgs.shape[1]==1)  #check the channel is 1
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	new_imgs = np.empty(imgs.shape)
	for i in range(imgs.shape[0]):
		new_imgs[i,0] = cv.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
	return new_imgs

def clahe_equalized(imgs):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


def o_distance(point_A,point_B):
    distance = math.sqrt((point_A[0][0]-point_B[0][0])**2 +(point_A[0][1]-point_B[0][1])**2)
    return distance
def find_centerline_flag_0(centerline_flag):
    point_num = centerline_flag.shape[0]
    for i in range(point_num):
        if centerline_flag[i]==0:
            return False, i
    return True, 0

def vector_norm_f(vector_org):
	vector0 = vector_org[0]
	vector1 = vector_org[1]

	vector_norm = [0,0]
	for i in range(2):
		vector_norm[i] = vector_org[i] / np.sqrt(vector0 ** 2 + vector1 ** 2 )
	return vector_norm

def up_sample_swc_rescale(swc_dir, length_threshold, resize_radio):
	swc_tree_upsample = read_swc_tree(swc_dir)
	swc_tree_upsample.rescale([resize_radio,resize_radio,resize_radio,resize_radio])
	swc_tree_upsample = up_sample_swc_tree(swc_tree_upsample, length_threshold)
	swc_tree_upsample.sort_node_list(key="default")

	return swc_tree_upsample

def down_sample_swc(swc_dir):
	swc_tree = read_swc_tree(swc_dir)
	swc_tree_downsample = down_sample_swc_tree(swc_tree)
	swc_tree_downsample.sort_node_list(key="default")

	return swc_tree_downsample

def get_centerline_swc(swc_dir, r):
	swc_tree_centerline = read_swc_tree(swc_dir)
	swc_node_list = swc_tree_centerline.get_node_list()
	for node in swc_node_list:
	    if node.is_virtual():
	        continue
	    node.set_r(r=r)
	return swc_tree_centerline

def get_centerline_direction_2d(resample_tree_data):
	centerline_direction = np.zeros([resample_tree_data.shape[0], 2], dtype=np.float32)
	for i in range(resample_tree_data.shape[0]):
		node_id = resample_tree_data[i][0]
		node_id_p = resample_tree_data[i][6]

		if node_id_p == -1:  # parent 
			if node_id != resample_tree_data.shape[0]:
				node_A = int(node_id) - 1
				node_B = int(node_id + 1) - 1
			else:
				node_A = int(node_id) - 1
				node_B = int(node_id) - 1
		else:
			node_son = int(node_id + 1) - 1

			if node_id != resample_tree_data.shape[0]:
				if resample_tree_data[node_son][6] == node_id:
					node_A = int(node_id) - 1
					node_B = int(node_id_p) - 1
				else:
					node_A = int(node_id) - 1
					node_B = int(node_id_p) - 1
			else:
				node_A = int(node_id) - 1
				node_B = int(node_id_p) - 1

		centerline_direction[i][0] = resample_tree_data[node_A][3] - resample_tree_data[node_B][3]
		centerline_direction[i][1] = resample_tree_data[node_A][2] - resample_tree_data[node_B][2]

	return centerline_direction

def get_centerline_circle_2d(centerline_sample_A, centerline_direction, r=1.0):
	circle_num = centerline_sample_A.shape[0]

	theta = np.arange(0.001, 2 * np.pi, 1 / (r + 0.1))

	circle_x = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)
	circle_y = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)
	circle_z = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)

	circle_vector = np.zeros([circle_num, 2], dtype=np.float32)


	u_vector = np.zeros([circle_num, 2], dtype=np.float32)

	for i in range(circle_num):
		# 获取centerline, norm_vector
		vector_x = centerline_direction[i][0]
		vector_y = centerline_direction[i][1]

		circle_vector[i][0] = vector_y / np.sqrt(vector_x ** 2 + vector_y ** 2)
		circle_vector[i][1] = vector_x / np.sqrt(vector_x ** 2 + vector_y ** 2)

		##############################################
	
		norm_vector_x = centerline_direction[i][0]
		norm_vector_y = centerline_direction[i][1]

		# 获取法向量u
		u_x = - norm_vector_y
		u_y = norm_vector_x


		# 获取u,v的单位向量
		u_n = math.sqrt(u_x ** 2 + u_y ** 2 + 1e-5)
		u_x_tilde = u_x / u_n
		u_y_tilde = u_y / u_n

		u_vector[i][0] = u_x_tilde
		u_vector[i][1] = u_y_tilde

	return circle_vector, u_vector

def gen_circle_gaussian_3d(size=[16,64,64], r=2.0, z_offset=0, x_offset=0, y_offset=0):
    z0 = size[0] // 2
    x0 = size[1] // 2
    y0 = size[2] // 2
    x0 += x_offset
    y0 += y_offset
    z0 += z_offset
    z, x, y = np.ogrid[:size[0], :size[1], :size[2]]

    image = 1 * np.exp(
        -(((x - x0) ** 2 / (2 * r ** 2)) + ((y - y0) ** 2 / (2 * r ** 2)) + ((z - z0) ** 2 / (2 * r ** 2))))

    return image * 255

def gen_circle_gaussian_2d(size=64, r=2.0, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += -y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    image = 1*np.exp(-(((x-x0)**2 /(2*r**2)) + ((y-y0)**2 /(2*r**2))))#*2 - 1



    return image*255

def prepare_train_datasets_2d(image_seq_dir, swc_tree_centerline_matirx, img_org, img_lab, img_skl, img_skl_dis, swc_tree_centerline_matirx_vector,swc_tree_centerline_matirx_u_vector, BATCH_SHAPE, image_name, PATCH_NUM, seq_len=1, img_gap=1):
	image_seq_dir = image_seq_dir + image_name + '/'
	if not os.path.exists(image_seq_dir):
		os.mkdir(image_seq_dir)

	img_shape = img_org.shape
	img_org_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1], 3])
	img_skl_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1]])
	img_skl_dis_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1]])

	z_half = 1
	x_half = BATCH_SHAPE[0]//2
	y_half = BATCH_SHAPE[1]//2

	img_org_temp[2*BATCH_SHAPE[0]:2*BATCH_SHAPE[0] + img_shape[0], 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], :] = copy.deepcopy(img_org)
	img_skl_temp[2*BATCH_SHAPE[0]:2*BATCH_SHAPE[0] + img_shape[0], 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1]] = copy.deepcopy(img_skl)
	img_skl_dis_temp = copy.deepcopy(img_skl_dis)


	swc_tree_node_flag = np.zeros([swc_tree_centerline_matirx.shape[0]])

	# 提取每条枝干的id
	branch_list = []
	for i in range(swc_tree_centerline_matirx.shape[0]):
		node_id_temp = swc_tree_centerline_matirx.shape[0] - 1 - i
		branch_list_temp = []
		branch_list_temp.append(node_id_temp)

		while node_id_temp != -2 and swc_tree_node_flag[int(node_id_temp)] == 0:
			swc_tree_node_flag[int(node_id_temp)] = 1
			node_id_temp = swc_tree_centerline_matirx[int(node_id_temp)][6] - 1
			branch_list_temp.append(node_id_temp)
		if len(branch_list_temp) > seq_len*2+1:
			branch_list.append(branch_list_temp)
			#print(len(branch_list_temp))
	print('image id:%s branch num: %d' % (image_name, len(branch_list)))

	branch_num = len(branch_list)


	for i in range(PATCH_NUM):
		if i%20==0:
			print('image id:%s %d / %d'% (image_name, i+1, PATCH_NUM))

		branch_id_rand = random.randint(0,branch_num-1)
		branch_temp_list = branch_list[branch_id_rand]
		branch_length = len(branch_temp_list)

		seed_id_rand = random.randint(seq_len,branch_length-1-seq_len)
		
		node0_2_id = branch_temp_list[seed_id_rand-3]
		node0_1_id = branch_temp_list[seed_id_rand-2]
		node0_id = branch_temp_list[seed_id_rand-1]
		node1_id = branch_temp_list[seed_id_rand]
		node2_id = branch_temp_list[seed_id_rand+1]
		node3_id = branch_temp_list[seed_id_rand+2]
		node4_id = branch_temp_list[seed_id_rand+3]
		node5_id = branch_temp_list[seed_id_rand+4]

		# norm
		image_seq_single_dir = image_seq_dir + '/' + str(i+1)  + '_pos_0/'
		if os.path.exists(image_seq_single_dir):
			rmtree(image_seq_single_dir)
			os.mkdir(image_seq_single_dir)
		else:
			os.mkdir(image_seq_single_dir)

		image_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], 3])
		image_lab_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1]])
		image_skl_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1]])
		image_img_dis_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1]])
		image_skl_dis_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1]])
		image_dis_branch_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1]])

		node_img_temp_dir = image_seq_single_dir + 'node_img.tif'
		node_img_lab_temp_dir = image_seq_single_dir + 'node_lab.tif'		
		node_img_skl_temp_dir = image_seq_single_dir + 'node_skl.tif'

		node_img_skl_dis_temp_dir = image_seq_single_dir + 'node_skl_dis.tif'
		node_img_skl_branch_temp_dir = image_seq_single_dir + 'node_skl_b.tif'
  
		node_img_walk_temp_dir = image_seq_single_dir + 'node_walk.tif'
		node_img_point_temp_dir = image_seq_single_dir + 'node_point.tif'
		node_img_edge_temp_dir = image_seq_single_dir + 'node_edge.tif'

  
  
		node_pos_float = swc_tree_centerline_matirx[int(node1_id)][2:4]
		node_pos = [int(round(j,0)) for j in node_pos_float]

		node_pos_p_float = swc_tree_centerline_matirx[int(node1_id-1)][2:4]
		node_pos_p = [int(round(j,0)) for j in node_pos_p_float]
		node_pos_s_float = swc_tree_centerline_matirx[int(node1_id+1)][2:4]
		node_pos_s = [int(round(j,0)) for j in node_pos_s_float]
  
		# 
		node_vector = node_pos_p_float - node_pos_s_float
		node_vector = vector_norm_f(node_vector)
		node_vector_u = [-node_vector[1], node_vector[0]]
		node_vector_u_n = np.linalg.norm(np.array(node_vector_u))
		node_u = node_vector_u / node_vector_u_n

		# enhanced
		data_enhanced = np.random.uniform(0,1)
		if data_enhanced > 0.5:
			x_rand_temp = random.uniform(-3, 3)
		else:
			x_rand_temp = 0
		node_rand_float = [0,0]
		node_rand_float[0] = x_rand_temp * node_u[0] 
		node_rand_float[1] = x_rand_temp * node_u[1] 
		node_rand = [int(round(rad_num,0)) for rad_num in node_rand_float]
		node_pos_float_positive = [a+b for a,b in zip(node_pos_float, node_rand_float)] 
		node_pos = [int(round(j,0)) for j in node_pos_float_positive]
  
  
		# get label
		node_img = copy.deepcopy(img_org_temp[node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half, :])
		node_skl = copy.deepcopy(img_skl_temp[node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half])
		node_skl_dis = copy.deepcopy(img_skl_dis_temp[node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half])

		image_stack[0] = copy.deepcopy(node_img)
		image_skl_stack[0] = copy.deepcopy(node_skl)
		image_skl_dis_stack[0] = copy.deepcopy(node_skl_dis)

		save_tif(image_stack, node_img_temp_dir, np.uint8)
		save_tif(image_skl_stack, node_img_skl_temp_dir, np.uint8)
		save_tif(image_skl_dis_stack, node_img_skl_dis_temp_dir, np.uint8)

		# walked path enhanced
		r = 1.0
		node_len_size = np.random.choice([2, 3, 4, 5], p=[0.1, 0.1, 0.1, 0.7])
		if node_len_size==2:
			node_id_list = [node0_id, node2_id]
		elif node_len_size==3:
			node_id_list = [node0_id, node2_id, node3_id]
		elif node_len_size==4:
			node_id_list = [node0_id, node2_id, node3_id, node4_id]
		else:
			node_id_list = [node0_id, node2_id, node3_id, node4_id, node5_id]
		node_pos_float_metric = np.zeros([len(node_id_list),2])
		vector_node_metric = np.zeros([len(node_id_list),2])
		for i_id in range(node_pos_float_metric.shape[0]):
			node_pos_float_metric[i_id] = swc_tree_centerline_matirx[int(node_id_list[i_id])][2:4]

		vector_node_metric[0] = (node_pos_float_metric[0] - node_pos_float_positive)
		for i_id in range(1,vector_node_metric.shape[0]):
			if i_id == 4:
				vector_node_metric[i_id] = (node_pos_float_metric[i_id] - node_pos_float_positive)*3 + [BATCH_SHAPE[0]//2,BATCH_SHAPE[1]//2]
			else:
				vector_node_metric[i_id] = (node_pos_float_metric[i_id] - node_pos_float_positive) + [BATCH_SHAPE[0]//2,BATCH_SHAPE[1]//2]
		
  
		
		mark = np.zeros((1, BATCH_SHAPE[0], BATCH_SHAPE[1]))
		mark_shape = (1, BATCH_SHAPE[0], BATCH_SHAPE[1])
		start_point = [y_half, x_half, 0]  
		start_node = Vertex(0, 0, 0, start_point[1], start_point[0], r, -1)

		end_node_list = []
		for i_id in range(1,vector_node_metric.shape[0]):
			end_node_temp = Vertex(1, 0, 0, vector_node_metric[i_id][1], vector_node_metric[i_id][0], r, -1)
			end_node_list.append(end_node_temp)
		
		setMarkWithCone(mark, Cone(Point3D(*start_node.pos), start_node.r, Point3D(*end_node_list[0].pos), end_node_list[0].r), mark_shape)
		for i_id in range(0,len(end_node_list)-1):
			setMarkWithSphere(mark, Sphere(Point3D(*end_node_list[i_id].pos), end_node_list[i_id].r), mark_shape)
			setMarkWithCone(mark, Cone(Point3D(*end_node_list[i_id].pos), end_node_list[i_id].r, Point3D(*end_node_list[i_id+1].pos), end_node_list[i_id+1].r), mark_shape)
		img_walk = mark.astype(np.uint8)
   
		
		future_node_id_list = [node0_id, node0_1_id, node0_2_id]
		future_node_pos_float_metric = np.zeros([len(future_node_id_list),2])
		future_vector_node_metric = np.zeros([len(future_node_id_list),2])
		img_edge_seq = np.zeros([len(future_node_id_list),BATCH_SHAPE[0], BATCH_SHAPE[1]])
		img_point_seq = np.zeros([len(future_node_id_list),BATCH_SHAPE[0], BATCH_SHAPE[1]])
  
		for i_id in range(future_node_pos_float_metric.shape[0]):
			future_node_pos_float_metric[i_id] = swc_tree_centerline_matirx[int(future_node_id_list[i_id])][2:4]

		for i_id in range(future_vector_node_metric.shape[0]):
			future_vector_node_metric[i_id] = (future_node_pos_float_metric[i_id] - node_pos_float_positive) +  [BATCH_SHAPE[0]//2,BATCH_SHAPE[1]//2]
   

		mark_edge = np.zeros((1, BATCH_SHAPE[0], BATCH_SHAPE[1]))
		future_node_list = []
		for i_id in range(0, future_vector_node_metric.shape[0]):
			future_node_temp = Vertex(1, 0, 0, future_vector_node_metric[i_id][1], future_vector_node_metric[i_id][0], r, -1)
			future_node_list.append(future_node_temp)
  
		setMarkWithSphere(mark_edge, Sphere(Point3D(*start_node.pos), start_node.r), mark_shape)
		setMarkWithCone(mark_edge, Cone(Point3D(*start_node.pos), start_node.r, Point3D(*future_node_list[0].pos), future_node_list[0].r), mark_shape)
		setMarkWithSphere(mark_edge, Sphere(Point3D(*future_node_list[0].pos), future_node_list[0].r), mark_shape)
		img_edge = copy.deepcopy(mark_edge.astype(np.uint8))
  
		for i_id in range(0,len(future_node_list)-1):
			setMarkWithSphere(mark_edge, Sphere(Point3D(*future_node_list[i_id].pos), future_node_list[i_id].r), mark_shape)
			setMarkWithCone(mark_edge, Cone(Point3D(*future_node_list[i_id].pos), future_node_list[i_id].r, Point3D(*future_node_list[i_id+1].pos), future_node_list[i_id+1].r), mark_shape)
		
		img_edge_b = copy.deepcopy(mark_edge.astype(np.uint8))
		img_edge_b_ = img_walk + img_edge_b
		img_edge_b_[img_edge_b_>255]=255
		save_tif(img_edge_b_, node_img_skl_branch_temp_dir, np.uint8)
  
  

		img_point = gen_circle_gaussian_3d(size =[1,64,64], z_offset=0, x_offset=future_vector_node_metric[0][1]-BATCH_SHAPE[1]//2, y_offset=future_vector_node_metric[0][0]-BATCH_SHAPE[0]//2).astype(np.uint8)
  
  
		data_enhanced = np.random.uniform(0,1)
		if data_enhanced > 0.9: 
			img_walk_ = np.zeros_like(img_walk)
			setMarkWithSphere(img_walk_, Sphere(Point3D(*start_node.pos), start_node.r), mark_shape)
			img_walk_ = img_walk_.astype(np.uint8)
			save_tif(img_walk_, node_img_walk_temp_dir, np.uint8)

			img_point = gen_circle_gaussian_3d(size =[1,64,64], z_offset=0, x_offset=-vector_node_metric[0][1], y_offset=-vector_node_metric[0][0]).astype(np.uint8)
			img_point_ = gen_circle_gaussian_3d(size =[1,64,64], z_offset=0, x_offset=vector_node_metric[1][1]-BATCH_SHAPE[1]//2, y_offset=vector_node_metric[1][0]-BATCH_SHAPE[0]//2).astype(np.uint8)
   
			img_point = img_point + img_point_
			img_point[img_point>255]=255
			save_tif(img_point, node_img_point_temp_dir, np.uint8)
			
   
	
			mark_edge = np.zeros((1, BATCH_SHAPE[0], BATCH_SHAPE[1]))
			neigh_node_temp_1 = Vertex(1, 0, 0, vector_node_metric[0][1]+BATCH_SHAPE[1]//2, vector_node_metric[0][0]+BATCH_SHAPE[0]//2, r, -1)
			neigh_node_temp_2 = Vertex(2, 0, 0, vector_node_metric[1][1], vector_node_metric[1][0], r, -1)

			setMarkWithCone(mark_edge, Cone(Point3D(*neigh_node_temp_1.pos), neigh_node_temp_1.r, Point3D(*start_node.pos), start_node.r), mark_shape)
			setMarkWithCone(mark_edge, Cone(Point3D(*start_node.pos), start_node.r, Point3D(*neigh_node_temp_2.pos), neigh_node_temp_2.r), mark_shape)
			setMarkWithSphere(mark_edge, Sphere(Point3D(*neigh_node_temp_2.pos), neigh_node_temp_2.r), mark_shape)
			setMarkWithSphere(mark_edge, Sphere(Point3D(*neigh_node_temp_1.pos), neigh_node_temp_1.r), mark_shape)
			img_edge = mark_edge.astype(np.uint8)
			save_tif(img_edge, node_img_edge_temp_dir, np.uint8)
		else:
			save_tif(img_walk, node_img_walk_temp_dir, np.uint8)
			save_tif(img_point, node_img_point_temp_dir, np.uint8)
			save_tif(img_edge, node_img_edge_temp_dir, np.uint8)
  
  
		# neg sample
		data_enhanced = np.random.uniform(0,1)
		if data_enhanced > 0.5: 
			image_seq_single_dir = image_seq_dir + '/' + str(i+1) + '_neg_0/'
			if os.path.exists(image_seq_single_dir):
				rmtree(image_seq_single_dir)
				os.mkdir(image_seq_single_dir)
			else:
				os.mkdir(image_seq_single_dir)
			
			image_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], 3])
			image_skl_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1]])
			image_skl_dis_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1]])

			node_img_temp_dir = image_seq_single_dir + 'node_img.tif'
			node_img_lab_temp_dir = image_seq_single_dir + 'node_lab.tif'
			node_img_skl_temp_dir = image_seq_single_dir + 'node_skl.tif'
			node_img_skl_dis_temp_dir = image_seq_single_dir + 'node_skl_dis.tif'
			node_img_skl_branch_temp_dir = image_seq_single_dir + 'node_skl_b.tif'
   
			node_img_walk_temp_dir = image_seq_single_dir + 'node_walk.tif'
			node_img_point_temp_dir = image_seq_single_dir + 'node_point.tif'
			node_img_edge_temp_dir = image_seq_single_dir + 'node_edge.tif'
			

			x_rand_temp = random.randint(BATCH_SHAPE[0], img_org_temp.shape[0]-BATCH_SHAPE[0])
			y_rand_temp = random.randint(BATCH_SHAPE[1], img_org_temp.shape[1]-BATCH_SHAPE[1])

			node_img_neg = copy.deepcopy(img_org_temp[x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half, :])
			node_skl_neg = copy.deepcopy(img_skl_temp[x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half])
			node_skl_dis_neg = copy.deepcopy(img_skl_dis_temp[x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half])
   


			image_stack[0] = copy.deepcopy(node_img_neg)
			image_skl_stack[0] = copy.deepcopy(node_skl_neg)
			image_skl_dis_stack[0] = copy.deepcopy(node_skl_dis_neg)

			save_tif(image_stack, node_img_temp_dir, np.uint8)
			save_tif(image_skl_stack, node_img_skl_temp_dir, np.uint8)
			save_tif(image_skl_dis_stack, node_img_skl_dis_temp_dir, np.uint8)


			img_walk = np.zeros_like(image_skl_stack)
			img_point = np.zeros_like(image_skl_stack)
			img_edge = np.zeros_like(image_skl_stack)
			img_edge_b = np.zeros_like(image_skl_stack)
			save_tif(img_walk, node_img_walk_temp_dir, np.uint8)
			save_tif(img_point, node_img_point_temp_dir, np.uint8)
			save_tif(img_edge, node_img_edge_temp_dir, np.uint8)
			save_tif(img_edge_b, node_img_skl_branch_temp_dir, np.uint8)
  



def main_training_data(input_dir):
	args = parse_args()
	datasets_name = args.datasets_name
	image_seq_dir = args.train_dataset_root_dir + 'training_datasets/'

	org_image_train_dir = args.image_dir + datasets_name + '/training/images_color/'
	org_mask_train_dir = args.image_dir + datasets_name + '/training/mask/'
	org_swc_train_dir = args.image_dir + datasets_name + '/training/swc/'

	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'
	temp_mask_tif_dir = args.image_dir + datasets_name + '/temp/mask/'

	temp_image_dis_tif_dir = args.image_dir + datasets_name + '/temp/images_dis/'
	temp_skl_dis_tif_dir = args.image_dir + datasets_name + '/temp/centerlines_dis/'
	
	batch_size = args.input_dim
	BATCH_SHAPE = [batch_size[0],batch_size[1]]

	image_name = input_dir.split("/")[-1].split(".")[0]

	if datasets_name == 'DRIVE':
		patch_num = args.N_patches // 20 
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '.swc'

		resize_radio = 2.0
		swc_upsample_length = 10.0
		sample_gap = 1
		seq_len = 5
	elif datasets_name == 'CHASEDB1':
		patch_num = args.N_patches // 20 // 5
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '_manual1.swc'

		resize_radio = 1.5
		swc_upsample_length = 10.0
		sample_gap = 1
		seq_len = 5
	elif datasets_name == 'ROAD':
		patch_num = args.N_patches // 800 
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '.swc'

		resize_radio = 1.0
		swc_upsample_length = 10.0
		sample_gap = 1
		seq_len = 5
	else:
		pause


	# 
	img_resize_dir = temp_image_tif_dir + image_name + '_img_resize.tif'
	img_color = open_tif(img_dir).astype(np.float32)
	img_new = transform.resize(img_color,(round(img_color.shape[0]*resize_radio), round(img_color.shape[1]*resize_radio)))
	save_tif(img_new, img_resize_dir, np.uint8)


	# mask
	mask_resize_dir = temp_mask_tif_dir + image_name + '_mask_resize.tif'
	img_mask = open_tif(img_mask_dir).astype(np.float32)
	img_mask = transform.resize(img_mask,(round(img_color.shape[0]*resize_radio), round(img_color.shape[1]*resize_radio)))
	save_tif(img_mask, mask_resize_dir, np.uint8)

	# swc，centerline
	swc_tree_upsample = up_sample_swc_rescale(swc_dir, swc_upsample_length, resize_radio)
	data_swc_upsample_dir_tmp = temp_swc_centerline_dir + image_name + '.upsample.swc'
	swc_save(swc_tree_upsample, data_swc_upsample_dir_tmp)
			
	# 
	swc_tree_centerline = get_centerline_swc(data_swc_upsample_dir_tmp, 1.0)
	data_swc_centerline_dir_tmp = temp_swc_centerline_dir + image_name + '.centerline.swc'
	swc_save(swc_tree_centerline, data_swc_centerline_dir_tmp)
	swc_save_preorder(data_swc_centerline_dir_tmp, data_swc_centerline_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_centerline_dir_tmp) # read swc

	img_skl = save_swc2tif(data_swc_centerline_dir_tmp, [1, img_new.shape[0], img_new.shape[1]])
	data_image_skl_dir_tmp = temp_centerline_dir + image_name + '_centerline_resize.tif'
	save_tif(img_skl, data_image_skl_dir_tmp, np.uint8)

	# 
	img_shape = img_new.shape
	img_skl_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1]])
	img_skl_temp[2*BATCH_SHAPE[0]:2*BATCH_SHAPE[0] + img_shape[0], 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1]] = copy.deepcopy(img_skl[0])
	# 
	data_skl_dis_dir_tmp = temp_skl_dis_tif_dir + image_name + '.skl_dis.tif'
	img_skl_f = np.ones_like(img_skl_temp) - img_skl_temp//255
	from scipy import ndimage as ndi
	img_skl_dis = ndi.distance_transform_edt(img_skl_f)
	img_skl_dis = img_skl_dis * 10 

	img_skl_dis[img_skl_dis>255]=255
	save_tif(img_skl_dis[2*BATCH_SHAPE[0]:-2*BATCH_SHAPE[0], 2*BATCH_SHAPE[1]:-2*BATCH_SHAPE[1]], data_skl_dis_dir_tmp, np.uint8)
 
	# exist
	swc_tree_exist = get_centerline_swc(data_swc_upsample_dir_tmp, 2.0)
	data_swc_exist_dir_tmp = temp_swc_centerline_dir + image_name + '.exist.swc'
	swc_save(swc_tree_exist, data_swc_exist_dir_tmp)
	swc_save_preorder(data_swc_exist_dir_tmp, data_swc_exist_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_exist_dir_tmp) # read swc

	label_new = save_swc2tif(data_swc_exist_dir_tmp, [1, img_new.shape[0], img_new.shape[1]])
	data_image_exist_dir_tmp = temp_centerline_dir + image_name + '_exist_resize.tif'
	save_tif(label_new, data_image_exist_dir_tmp, np.uint8)

	# # 
	swc_tree_centerline_matirx = read_swc_tree_matrix(data_swc_upsample_dir_tmp)

	swc_tree_centerline_matirx_radius = np.zeros([swc_tree_centerline_matirx.shape[0], 1])
	for i in range(swc_tree_centerline_matirx.shape[0]):
		swc_tree_centerline_matirx_radius[i][0] = swc_tree_centerline_matirx[i][5]

	swc_tree_centerline_matirx_direction = get_centerline_direction_2d(swc_tree_centerline_matirx)
	swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector = get_centerline_circle_2d(swc_tree_centerline_matirx, swc_tree_centerline_matirx_direction)

	prepare_train_datasets_2d(image_seq_dir, swc_tree_centerline_matirx, img_new, label_new, img_skl, img_skl_dis, swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector, BATCH_SHAPE, image_name, patch_num, seq_len = seq_len, img_gap=sample_gap)

	time.sleep(1)

def main_test_data(input_dir):
	args = parse_args()
	datasets_name = args.datasets_name

	image_seq_dir = args.train_dataset_root_dir + 'test_datasets/'
	org_image_train_dir = args.image_dir + datasets_name + '/test/images_color/'
	org_mask_train_dir = args.image_dir + datasets_name + '/test/mask/'
	org_swc_train_dir = args.image_dir + datasets_name + '/test/swc/'

	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_mask_tif_dir = args.image_dir + datasets_name + '/temp/mask/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'
	

	temp_image_dis_tif_dir = args.image_dir + datasets_name + '/temp/images_dis/'
	temp_skl_dis_tif_dir = args.image_dir + datasets_name + '/temp/centerlines_dis/'
	
	batch_size = args.input_dim
	BATCH_SHAPE = [batch_size[0],batch_size[1]]
 
 
	image_name = input_dir.split("/")[-1].split(".")[0]

	if datasets_name == 'DRIVE':
		patch_num = args.N_patches // 20 // 10
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '.swc'

		resize_radio = 2.0
		swc_upsample_length = 99.0
		sample_gap = 1
		seq_len = 5
	elif datasets_name == 'CHASEDB1':
		patch_num = args.N_patches // 8 // 10 // 5
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '_manual1.swc'
		
		resize_radio = 1.5
		swc_upsample_length = 99.0
		sample_gap = 1
		seq_len = 5
	elif datasets_name == 'ROAD':
		patch_num = args.N_patches // 13 // 10
		img_dir = org_image_train_dir + image_name + '.tif'
		img_mask_dir = org_mask_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '.swc'
		
		resize_radio = 1.0
		swc_upsample_length = 10.0
		sample_gap = 1
		seq_len = 5
	else:
		pause

	# 
	img_resize_dir = temp_image_tif_dir + image_name + '_img_resize.tif'
	img_color = open_tif(img_dir).astype(np.float32)
	img_new = transform.resize(img_color,(round(img_color.shape[0]*resize_radio), round(img_color.shape[1]*resize_radio)))
	save_tif(img_new, img_resize_dir, np.uint8)
 
	# mask
	mask_resize_dir = temp_mask_tif_dir + image_name + '_mask_resize.tif'
	img_mask = open_tif(img_mask_dir).astype(np.float32)
	img_mask = transform.resize(img_mask,(round(img_color.shape[0]*resize_radio), round(img_color.shape[1]*resize_radio)))
	save_tif(img_mask, mask_resize_dir, np.uint8)

	# swc，centerline
	swc_tree_upsample = up_sample_swc_rescale(swc_dir, swc_upsample_length, resize_radio)
	data_swc_upsample_dir_tmp = temp_swc_centerline_dir + image_name + '.upsample.swc'
	swc_save(swc_tree_upsample, data_swc_upsample_dir_tmp)
			
	# 
	swc_tree_centerline = get_centerline_swc(data_swc_upsample_dir_tmp, 1.0)
	data_swc_centerline_dir_tmp = temp_swc_centerline_dir + image_name + '.centerline.swc'
	swc_save(swc_tree_centerline, data_swc_centerline_dir_tmp)
	swc_save_preorder(data_swc_centerline_dir_tmp, data_swc_centerline_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_centerline_dir_tmp) # read swc

	img_skl = save_swc2tif(data_swc_centerline_dir_tmp, [1, img_new.shape[0], img_new.shape[1]])
	data_image_skl_dir_tmp = temp_centerline_dir + image_name + '_centerline_resize.tif'
	save_tif(img_skl, data_image_skl_dir_tmp, np.uint8)
	
	# 
	img_shape = img_new.shape
	img_skl_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1]])
	img_skl_temp[2*BATCH_SHAPE[0]:2*BATCH_SHAPE[0] + img_shape[0], 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1]] = copy.deepcopy(img_skl[0])
	# 
	data_skl_dis_dir_tmp = temp_skl_dis_tif_dir + image_name + '.skl_dis.tif'
	img_skl_f = np.ones_like(img_skl_temp) - img_skl_temp//255
	from scipy import ndimage as ndi
	img_skl_dis = ndi.distance_transform_edt(img_skl_f)
	img_skl_dis = img_skl_dis * 10 

	img_skl_dis[img_skl_dis>255]=255
	save_tif(img_skl_dis[2*BATCH_SHAPE[0]:-2*BATCH_SHAPE[0], 2*BATCH_SHAPE[1]:-2*BATCH_SHAPE[1]], data_skl_dis_dir_tmp, np.uint8)
 
	# exist
	swc_tree_exist = get_centerline_swc(data_swc_upsample_dir_tmp, 2.0)
	data_swc_exist_dir_tmp = temp_swc_centerline_dir + image_name + '.exist.swc'
	swc_save(swc_tree_exist, data_swc_exist_dir_tmp)
	swc_save_preorder(data_swc_exist_dir_tmp, data_swc_exist_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_exist_dir_tmp) # read swc

	label_new = save_swc2tif(data_swc_exist_dir_tmp, [1, img_new.shape[0], img_new.shape[1]])
	data_image_exist_dir_tmp = temp_centerline_dir + image_name + '_exist_resize.tif'
	save_tif(label_new, data_image_exist_dir_tmp, np.uint8)

	# 
	swc_tree_centerline_matirx = read_swc_tree_matrix(data_swc_upsample_dir_tmp)

	swc_tree_centerline_matirx_radius = np.zeros([swc_tree_centerline_matirx.shape[0], 1])
	for i in range(swc_tree_centerline_matirx.shape[0]):
		swc_tree_centerline_matirx_radius[i][0] = swc_tree_centerline_matirx[i][5]

	swc_tree_centerline_matirx_direction = get_centerline_direction_2d(swc_tree_centerline_matirx)
	swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector = get_centerline_circle_2d(swc_tree_centerline_matirx, swc_tree_centerline_matirx_direction)

	prepare_train_datasets_2d(image_seq_dir, swc_tree_centerline_matirx, img_new, label_new, img_skl, img_skl_dis, swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector, BATCH_SHAPE, image_name, patch_num, seq_len = seq_len, img_gap=sample_gap)


if __name__ == '__main__':
	args = parse_args()
	
	datasets_name = args.datasets_name
	cpu_core_num = args.multi_cpu
	batch_size = args.input_dim
	patch_num = args.N_patches
 


	BATCH_SHAPE = [batch_size[0],batch_size[1]]


	print("loading " + datasets_name + " datasets")
	

	org_image_train_dir = args.image_dir + datasets_name + '/training/images_color/'
	org_label_train_dir = args.image_dir + datasets_name + '/training/labels/'
	org_mask_train_dir = args.image_dir + datasets_name + '/training/mask/'
	org_swc_train_dir = args.image_dir + datasets_name + '/training/swc/'

	org_image_test_dir = args.image_dir + datasets_name + '/test/images_color/'
	org_label_test_dir = args.image_dir + datasets_name + '/test/labels/'
	org_mask_test_dir = args.image_dir + datasets_name + '/test/mask/'
	org_swc_test_dir = args.image_dir + datasets_name + '/test/swc/'


	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_label_tif_dir = args.image_dir + datasets_name + '/temp/labels/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'

	training_datasets_dir = args.train_dataset_root_dir + 'training_datasets/'
	test_datasets_dir = args.train_dataset_root_dir + 'test_datasets/'

	if not os.path.exists(args.image_dir + datasets_name + '/temp'):
		os.makedirs(args.image_dir + datasets_name + '/temp')
	if not os.path.exists(temp_image_tif_dir):
		os.makedirs(temp_image_tif_dir)
	if not os.path.exists(temp_label_tif_dir):
		os.makedirs(temp_label_tif_dir)
	if not os.path.exists(temp_swc_centerline_dir):
		os.makedirs(temp_swc_centerline_dir)
	if not os.path.exists(temp_centerline_dir):
		os.makedirs(temp_centerline_dir)
		
	if not os.path.exists(args.train_dataset_root_dir):
		os.makedirs(args.train_dataset_root_dir)
	if not os.path.exists(training_datasets_dir):
		os.makedirs(training_datasets_dir)
	if not os.path.exists(test_datasets_dir):
		os.makedirs(test_datasets_dir)


	
	org_label_list = glob.glob(org_label_train_dir + '*.tif')
	org_label_num = len(org_label_list)
	print('find %d images' % (org_label_num))
	pool = mp.Pool(processes=cpu_core_num)  # we set cpu core is 4
	pool.map(main_training_data, org_label_list) # 

	org_label_list = glob.glob(org_label_test_dir + '*.tif')
	org_label_num = len(org_label_list)
	print('find %d images' % (org_label_num))
	pool = mp.Pool(processes=cpu_core_num)  
	pool.map(main_test_data, org_label_list)


	import shutil
	total_training_data_num = 0
	training_dataset_list = glob.glob(training_datasets_dir + '*/')
	training_dataset_image_num = len(training_dataset_list)
	print('find %d image folders' % (training_dataset_image_num))
	for training_dataset_image_dir in training_dataset_list:
		training_dataset_image_list = glob.glob(training_dataset_image_dir + '*/')
		training_dataset_image_patch_num = len(training_dataset_image_list)
		print('Folder: %s, find %d images patches' % (training_dataset_image_dir.split('/')[-2], training_dataset_image_patch_num))
		total_training_data_num += training_dataset_image_patch_num
		# for training_dataset_image_patch_dir in training_dataset_image_list:
		# 	shutil.rmtree(training_dataset_image_patch_dir)
	print('TOTAL %d images patches' % (total_training_data_num))

	total_test_data_num = 0
	test_dataset_list = glob.glob(test_datasets_dir + '*/')
	test_dataset_image_num = len(test_dataset_list)
	print('find %d image folders' % (test_dataset_image_num))
	for test_dataset_image_dir in test_dataset_list:
		test_dataset_image_list = glob.glob(test_dataset_image_dir + '*/')
		test_dataset_image_patch_num = len(test_dataset_image_list)
		print('Folder: %s, find %d images patches' % (test_dataset_image_dir.split('/')[-2], test_dataset_image_patch_num))
		total_test_data_num += test_dataset_image_patch_num
		# for test_dataset_image_patch_dir in test_dataset_image_list:
		# 	shutil.rmtree(test_dataset_image_patch_dir)
	print('TOTAL %d images patches' % (total_test_data_num))



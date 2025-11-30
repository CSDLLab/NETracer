import numpy as np
from lib.klib.baseio import *
from scipy import ndimage
from scipy.ndimage import filters as ndfilter
from scipy.ndimage import morphology, generate_binary_structure

from lib.klib.glib.DrawSimulationSWCModel import simulate3DTreeModel_dendrite, save_swc, setMarkWithSphere, setMarkWithCone
from lib.klib.glib.Obj3D import Point3D, Sphere, Cone
from lib.klib.glib.SWCExtractor import Vertex

from lib.swclib.swc_io import swc_save, swc_save_preorder, read_swc_tree, read_swc_tree_matrix, swc_save_metric
from lib.swclib.swc_tree import SwcTree
from lib.swclib.swc_node import SwcNode
from lib.swclib.re_sample import up_sample_swc_tree, down_sample_swc_tree

from lib.swclib import euclidean_point
from lib.swclib import edge_match_utils, point_match_utils

from lib.gwdt import gwdt

import copy
import cv2 as cv
import multiprocessing as mp
# from skimage import morphology, transform
import queue
import time
import math
import random
import tifffile
# import GeodisTK
import os
from shutil import rmtree
from PIL import Image
import sys
sys.setrecursionlimit(100000)

import argparse

from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# python prepare_datasets_3D.py --datasets_name FMOST --train_dataset_root_dir /home/xxx/datasets/FMOST/training_data_v5_1_1/

def parse_args():
	parser = argparse.ArgumentParser()

	# (input dir) orginal data
	parser.add_argument('--datasets_name', default='FMOST',help='datasets name') # CHASEDB1

	parser.add_argument('--image_dir', default='/8T1/xxx/', help='orginal image saved here')
	
	# (output dir)
	parser.add_argument('--train_dataset_root_dir', default='/4T/xxx/deepneutracing/vecroad_3d/FMOST/training_data/',help='orginal centerline saved here')
	parser.add_argument('--N_patches', default=20000,help='Number of training image patches') # 150000
	parser.add_argument('--data_type', default='uint16') # 80000 20000

	parser.add_argument('--input_dim', type=int, default=(16,64,64))
	parser.add_argument('--multi_cpu', type=int, default=5)

	args = parser.parse_args()
	
	return args


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
	vector2 = vector_org[2]

	vector_norm = [0,0,0]
	for i in range(3):
		vector_norm[i] = vector_org[i] / np.sqrt(vector0 ** 2 + vector1 ** 2 + vector2 ** 2)
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

def in_range(n, start, end = 0):
    return start <= n <= end if end >= start else end <= n <= start

def countRegion(img, seed, threshold = 128.0):
    mask = np.zeros_like(img)
    structure = generate_binary_structure(img.ndim, 3)

    node_pool = []
    node_pool.append(seed)
    while len(node_pool) != 0:
        current_node_pos = node_pool.pop()

        z_temp = current_node_pos[0]
        x_temp = current_node_pos[1]
        y_temp = current_node_pos[2]

        if img[z_temp][x_temp][y_temp] < threshold:
            mask[z_temp][x_temp][y_temp] = 255

            structure_temp = copy.deepcopy(
                img[z_temp - 1:z_temp + 2, x_temp - 1:x_temp + 2, y_temp - 1:y_temp + 2]) * structure

            next_node_list_temp = np.where(structure_temp<=threshold)

            for node_temp_id in range(len(next_node_list_temp[0])):
                z_ = next_node_list_temp[0][node_temp_id] + z_temp - 1
                x_ = next_node_list_temp[1][node_temp_id] + x_temp - 1
                y_ = next_node_list_temp[2][node_temp_id] + y_temp - 1

                next_node_pos = [z_, x_, y_]


                if mask[z_][x_][y_] != 255:
                    if in_range(z_, 1, img.shape[0]-2) and in_range(x_, 1, img.shape[1]-2) and in_range(y_, 1, img.shape[2]-2) :
                        node_pool.append(next_node_pos)

    return mask

def get_centerline_circle(centerline_sample_A, centerline_direction, r=1.0):
	circle_num = centerline_sample_A.shape[0]

	theta = np.arange(0.001, 2 * np.pi, 1 / (r + 0.1))

	circle_x = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)
	circle_y = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)
	circle_z = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)

	circle_vector = np.zeros([circle_num, 3], dtype=np.float32)
	# circle_angle = np.zeros([circle_num, 2], dtype=np.float32)

	u_vector = np.zeros([circle_num, 3], dtype=np.float32)
	v_vector = np.zeros([circle_num, 3], dtype=np.float32)

	for i in range(circle_num):
		# centerline, norm_vector
		vector_y = centerline_direction[i][0]
		vector_x = centerline_direction[i][1]
		vector_z = centerline_direction[i][2]

		circle_vector[i][0] = vector_z / np.sqrt(vector_x ** 2 + vector_y ** 2 + vector_z ** 2)
		circle_vector[i][1] = vector_x / np.sqrt(vector_x ** 2 + vector_y ** 2 + vector_z ** 2)
		circle_vector[i][2] = vector_y / np.sqrt(vector_x ** 2 + vector_y ** 2 + vector_z ** 2)

		##############################################
	
		norm_vector_x = centerline_direction[i][0]
		norm_vector_y = centerline_direction[i][1]
		norm_vector_z = centerline_direction[i][2]

		# 
		u_x = norm_vector_y
		u_y = - norm_vector_x
		u_z = 1e-7

		# 
		v_x = norm_vector_x * norm_vector_z
		v_y = norm_vector_y * norm_vector_z
		v_z = - norm_vector_x ** 2 - norm_vector_y ** 2 + 1e-7

		#
		u_n = math.sqrt(u_x ** 2 + u_y ** 2 + u_z ** 2)
		u_x_tilde = u_x / u_n
		u_y_tilde = u_y / u_n
		u_z_tilde = u_z / u_n

		v_n = math.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
		v_x_tilde = v_x / v_n
		v_y_tilde = v_y / v_n
		v_z_tilde = v_z / v_n

		u_vector[i][0] = u_x_tilde
		u_vector[i][1] = u_y_tilde
		u_vector[i][2] = u_z_tilde

		v_vector[i][0] = v_x_tilde
		v_vector[i][1] = v_y_tilde
		v_vector[i][2] = v_z_tilde

	return circle_vector, u_vector, v_vector

def get_centerline_direction(resample_tree_data):
	centerline_direction = np.zeros([resample_tree_data.shape[0], 3], dtype=np.float32)
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

		centerline_direction[i][0] = resample_tree_data[node_A][4] - resample_tree_data[node_B][4]
		centerline_direction[i][1] = resample_tree_data[node_A][3] - resample_tree_data[node_B][3]
		centerline_direction[i][2] = resample_tree_data[node_A][2] - resample_tree_data[node_B][2]

	return centerline_direction


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

def prepare_train_datasets(image_seq_dir, swc_tree_centerline_matirx, img_org, img_lab, img_skl, img_org_dis, img_skl_dis, swc_tree_centerline_matirx_radius, swc_tree_centerline_matirx_vector,swc_tree_centerline_matirx_u_vector, swc_tree_centerline_matirx_v_vector, BATCH_SHAPE, image_name, PATCH_NUM, seq_len=1, img_gap=1):
	args = parse_args()
	data_type_ = args.data_type
	if data_type_ == 'uint8':
		data_type = np.uint8
	else:
		data_type = np.uint16
 
	image_seq_dir = image_seq_dir + image_name + '/'
	if not os.path.exists(image_seq_dir):
		os.mkdir(image_seq_dir)

	img_shape = img_org.shape
	img_org_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1],img_shape[2]+4*BATCH_SHAPE[2]])
	img_skl_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1],img_shape[2]+4*BATCH_SHAPE[2]])

	z_half = BATCH_SHAPE[0]//2
	x_half = BATCH_SHAPE[1]//2
	y_half = BATCH_SHAPE[2]//2
	
	img_org_temp[2*BATCH_SHAPE[0]:2*BATCH_SHAPE[0] + img_shape[0], 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], 2*BATCH_SHAPE[2]:2*BATCH_SHAPE[2] + img_shape[2]] = copy.deepcopy(img_org)
	img_skl_temp[2*BATCH_SHAPE[0]:2*BATCH_SHAPE[0] + img_shape[0], 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], 2*BATCH_SHAPE[2]:2*BATCH_SHAPE[2] + img_shape[2]] = copy.deepcopy(img_skl)
	
	img_skl_dis_temp = copy.deepcopy(img_skl_dis)

	swc_tree_node_flag = np.zeros([swc_tree_centerline_matirx.shape[0]])


	branch_list = []
	for i in range(swc_tree_centerline_matirx.shape[0]):
		node_id_temp = swc_tree_centerline_matirx.shape[0] - 1 - i
		branch_list_temp = []
		branch_list_temp.append(node_id_temp)

		while node_id_temp != -2 and swc_tree_node_flag[int(node_id_temp)] == 0:
			swc_tree_node_flag[int(node_id_temp)] = 1
			node_id_temp = swc_tree_centerline_matirx[int(node_id_temp)][6] - 1
			branch_list_temp.append(int(node_id_temp))
		if len(branch_list_temp) > seq_len*2+1:
			branch_list.append(branch_list_temp)
	print('id:%s %d' % (image_name, len(branch_list)))

	
	branch_num = len(branch_list)

	for i in range(PATCH_NUM):
		if i%20==0:
			print('id:%s %d / %d'% (image_name, i+1, PATCH_NUM))

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



		# 
		image_seq_single_dir = image_seq_dir + '/' + str(i+1)  + '_pos_0/'
		if os.path.exists(image_seq_single_dir):
			rmtree(image_seq_single_dir)
			os.mkdir(image_seq_single_dir)
		else:
			os.mkdir(image_seq_single_dir)


		image_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])
		image_lab_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])
		image_skl_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])
		image_img_dis_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])
		image_skl_dis_stack_ = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])
		image_skl_dis_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])
		image_skl_branch_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])

		node_img_temp_dir = image_seq_single_dir + 'node_img.tif'
		node_img_lab_temp_dir = image_seq_single_dir + 'node_lab.tif'		
		node_img_skl_temp_dir = image_seq_single_dir + 'node_skl.tif'

		node_img_skl_dis_temp_dir = image_seq_single_dir + 'node_skl_dis.tif'
		
		node_img_skl_branch_temp_dir = image_seq_single_dir + 'node_skl_b.tif'
  
		node_img_walk_temp_dir = image_seq_single_dir + 'node_walk.tif'
		node_img_point_temp_dir = image_seq_single_dir + 'node_point.tif'
		node_img_edge_temp_dir = image_seq_single_dir + 'node_edge.tif'
		node_img_point_center_temp_dir = image_seq_single_dir + 'node_point_center.tif'


		node_pos_float = swc_tree_centerline_matirx[int(node1_id)][2:5]
		node_pos = [int(round(j,0)) for j in node_pos_float]

		node_pos_p_float = swc_tree_centerline_matirx[int(node0_id)][2:5]
		node_pos_p = [int(round(j,0)) for j in node_pos_p_float]
		node_pos_s_float = swc_tree_centerline_matirx[int(node2_id)][2:5]
		node_pos_s = [int(round(j,0)) for j in node_pos_s_float]
		
		#
		node_vector = node_pos_p_float - node_pos_s_float
		node_vector = vector_norm_f(node_vector)
		node_vector_u = [1e-7, -node_vector[2], node_vector[1]]
		node_vector_v = [- node_vector[1] ** 2 - node_vector[2] ** 2 + 1e-7, node_vector[1]*node_vector[0], node_vector[2]*node_vector[0]]
		node_vector_u_n = np.linalg.norm(np.array(node_vector_u))
		node_vector_v_n = np.linalg.norm(np.array(node_vector_v))
		node_u = node_vector_u / node_vector_u_n
		node_v = node_vector_v / node_vector_v_n

		data_enhanced = np.random.uniform(0,1)
		if data_enhanced > 0.5:
			x_rand_temp = random.uniform(-1, 1)
			y_rand_temp = random.uniform(-1, 1)
		else:
			x_rand_temp = 0
			y_rand_temp = 0
		node_rand_float = [0,0,0]
		node_rand_float[2] = x_rand_temp * node_u[0] + y_rand_temp * node_v[0]
		node_rand_float[0] = x_rand_temp * node_u[1] + y_rand_temp * node_v[1]
		node_rand_float[1] = x_rand_temp * node_u[2] + y_rand_temp * node_v[2]
		node_rand = [int(round(rad_num,0)) for rad_num in node_rand_float]
		node_pos_float_positive = [a+b for a,b in zip(node_pos_float, node_rand_float)] 
		node_pos = [int(round(j,0)) for j in node_pos_float_positive]


		node_img = copy.deepcopy(img_org_temp[node_pos[2]+ 3*z_half:node_pos[2] + 5*z_half, node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half])
		node_skl = copy.deepcopy(img_skl_temp[node_pos[2]+ 3*z_half:node_pos[2] + 5*z_half, node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half])
  
		node_skl_dis = copy.deepcopy(img_skl_dis_temp[node_pos[2]+ 3*z_half:node_pos[2] + 5*z_half, node_pos[1]+ 3*x_half:node_pos[1] + 5*x_half, node_pos[0]+ 3*y_half:node_pos[0] + 5*y_half])

		image_stack[0] = copy.deepcopy(node_img)

		image_skl_stack[0] = copy.deepcopy(node_skl)

		image_skl_dis_stack[0] = copy.deepcopy(node_skl_dis)


		save_tif(image_stack, node_img_temp_dir, data_type)
		save_tif(image_skl_stack, node_img_skl_temp_dir, np.uint8)
		save_tif(image_skl_dis_stack, node_img_skl_dis_temp_dir, np.uint8)

		

  

		node_len_size = np.random.choice([2, 3, 4, 5], p=[0.1, 0.1, 0.1, 0.7])
		if node_len_size==2:
			node_id_list = [node0_id, node2_id]
		elif node_len_size==3:
			node_id_list = [node0_id, node2_id, node3_id]
		elif node_len_size==4:
			node_id_list = [node0_id, node2_id, node3_id, node4_id]
		else:
			node_id_list = [node0_id, node2_id, node3_id, node4_id, node5_id]
		node_pos_float_metric = np.zeros([len(node_id_list),3])
		vector_node_metric = np.zeros([len(node_id_list),3])
		for i_id in range(node_pos_float_metric.shape[0]):
			node_pos_float_metric[i_id] = swc_tree_centerline_matirx[int(node_id_list[i_id])][2:5]

		vector_node_metric[0] = (node_pos_float_metric[0] - node_pos_float_positive)
		for i_id in range(1,vector_node_metric.shape[0]):
			if i_id == 4:
				vector_node_metric[i_id] = (node_pos_float_metric[i_id] - node_pos_float_positive)*3 + [BATCH_SHAPE[1]//2,BATCH_SHAPE[2]//2,BATCH_SHAPE[0]//2]
			else:
				vector_node_metric[i_id] = (node_pos_float_metric[i_id] - node_pos_float_positive) + [BATCH_SHAPE[1]//2,BATCH_SHAPE[2]//2,BATCH_SHAPE[0]//2]
		vector_node_edge = (node_pos_float_metric[0] - node_pos_float_positive) + [BATCH_SHAPE[1]//2,BATCH_SHAPE[2]//2,BATCH_SHAPE[0]//2]
  
		

		# 		
		mark = np.zeros((BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]))
		mark_edge = np.zeros((BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]))
		mark_shape = (BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2])



		# 定义直线的起点和终点坐标
		r = 1.0
		start_point = [y_half, x_half, z_half]  # 
		start_node = Vertex(0, 0, start_point[2], start_point[1], start_point[0], r, -1)

		end_node_list = []
		for i_id in range(1,vector_node_metric.shape[0]):
			end_node_temp = Vertex(1, 0, vector_node_metric[i_id][2], vector_node_metric[i_id][1], vector_node_metric[i_id][0], r, -1)
			end_node_list.append(end_node_temp)

		
		setMarkWithCone(mark, Cone(Point3D(*start_node.pos), start_node.r, Point3D(*end_node_list[0].pos), end_node_list[0].r), mark_shape)
		for i_id in range(0,len(end_node_list)-1):
			setMarkWithCone(mark, Cone(Point3D(*end_node_list[i_id].pos), end_node_list[i_id].r, Point3D(*end_node_list[i_id+1].pos), end_node_list[i_id+1].r), mark_shape)

		img_walk = mark.astype(np.uint8)
  

		future_node = Vertex(1, 0, vector_node_edge[2], vector_node_edge[1], vector_node_edge[0], r, -1)
		setMarkWithSphere(mark_edge, Sphere(Point3D(*start_node.pos), start_node.r), mark_shape)
		setMarkWithCone(mark_edge, Cone(Point3D(*start_node.pos), start_node.r, Point3D(*future_node.pos), future_node.r), mark_shape)
		setMarkWithSphere(mark_edge, Sphere(Point3D(*future_node.pos), future_node.r), mark_shape)
		img_edge = mark_edge.astype(np.uint8)
  

		img_point = gen_circle_gaussian_3d(z_offset=vector_node_metric[0][2], x_offset=vector_node_metric[0][1], y_offset=vector_node_metric[0][0]).astype(np.uint8)


		# 
		data_enhanced = np.random.uniform(0,1)
		if data_enhanced > 0.9: # 
			img_walk_ = np.zeros_like(img_walk)
			setMarkWithSphere(img_walk_, Sphere(Point3D(*start_node.pos), start_node.r), mark_shape)
			img_walk_ = img_walk_.astype(np.uint8)
			save_tif(img_walk_, node_img_walk_temp_dir, np.uint8)

			# img_point_ = gen_circle_gaussian_3d(z_offset=-vector_node_1_0[2], x_offset=-vector_node_1_0[1], y_offset=-vector_node_1_0[0]).astype(np.uint8)
			img_point = gen_circle_gaussian_3d(z_offset=vector_node_metric[0][2], x_offset=vector_node_metric[0][1], y_offset=vector_node_metric[0][0]).astype(np.uint8)
			img_point_ = gen_circle_gaussian_3d(z_offset=vector_node_metric[1][2]-BATCH_SHAPE[0]//2, x_offset=vector_node_metric[1][1]-BATCH_SHAPE[1]//2, y_offset=vector_node_metric[1][0]-BATCH_SHAPE[2]//2).astype(np.uint8)
   
			img_point = img_point + img_point_
			img_point[img_point>255]=255
			save_tif(img_point, node_img_point_temp_dir, np.uint8)
   
			# 
			mark_edge = np.zeros((BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]))
			neigh_node_temp_1 = Vertex(1, 0, vector_node_metric[0][2]+BATCH_SHAPE[0]//2, vector_node_metric[0][1]+BATCH_SHAPE[1]//2, vector_node_metric[0][0]+BATCH_SHAPE[2]//2, r, -1)
			neigh_node_temp_2 = Vertex(2, 0, vector_node_metric[1][2], vector_node_metric[1][1], vector_node_metric[1][0], r, -1)

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

		

		data_enhanced = np.random.uniform(0,1)
		if data_enhanced > 0.75: 
			image_seq_single_dir = image_seq_dir + '/' + str(i+1) + '_neg_0/'
			if os.path.exists(image_seq_single_dir):
				rmtree(image_seq_single_dir)
				os.mkdir(image_seq_single_dir)
			else:
				os.mkdir(image_seq_single_dir)
			
			image_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])
			image_skl_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])
			image_skl_dis_stack = np.zeros([1, BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2]])



			node_img_temp_dir = image_seq_single_dir + 'node_img.tif'
			node_img_lab_temp_dir = image_seq_single_dir + 'node_lab.tif'
			node_img_skl_temp_dir = image_seq_single_dir + 'node_skl.tif'
			node_img_org_dis_temp_dir = image_seq_single_dir + 'node_img_dis.tif'
			node_img_skl_dis_temp_dir = image_seq_single_dir + 'node_skl_dis.tif'
			node_img_dis_branch_temp_dir = image_seq_single_dir + 'node_dis_b.tif'

			node_img_walk_temp_dir = image_seq_single_dir + 'node_walk.tif'
			node_img_point_temp_dir = image_seq_single_dir + 'node_point.tif'
			node_img_edge_temp_dir = image_seq_single_dir + 'node_edge.tif'
			
			z_rand_temp = random.randint(BATCH_SHAPE[0], img_org_temp.shape[0]-BATCH_SHAPE[0])
			x_rand_temp = random.randint(BATCH_SHAPE[1], img_org_temp.shape[1]-BATCH_SHAPE[1])
			y_rand_temp = random.randint(BATCH_SHAPE[2], img_org_temp.shape[2]-BATCH_SHAPE[2])

			node_img_neg = copy.deepcopy(img_org_temp[z_rand_temp - z_half:z_rand_temp + z_half, x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half])

			node_skl_neg = copy.deepcopy(img_skl_temp[z_rand_temp - z_half:z_rand_temp + z_half, x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half])
			

			node_skl_dis_neg = copy.deepcopy(img_skl_dis_temp[z_rand_temp - z_half:z_rand_temp + z_half, x_rand_temp - x_half:x_rand_temp + x_half, y_rand_temp - y_half:y_rand_temp + y_half])
   
			


			image_stack[0] = copy.deepcopy(node_img_neg)
			image_skl_stack[0] = copy.deepcopy(node_skl_neg)
			image_skl_dis_stack[0] = copy.deepcopy(node_skl_dis_neg)
  
			save_tif(image_stack, node_img_temp_dir, data_type)
			save_tif(image_skl_stack, node_img_skl_temp_dir, np.uint8)

			save_tif(image_skl_dis_stack, node_img_skl_dis_temp_dir, np.uint8)


			img_walk = np.zeros_like(image_stack)
			img_point = np.zeros_like(image_stack)
			img_edge = np.zeros_like(image_stack)
			save_tif(img_walk, node_img_walk_temp_dir, np.uint8)
			save_tif(img_point, node_img_point_temp_dir, np.uint8)
			save_tif(img_edge, node_img_edge_temp_dir, np.uint8)

	return 0




def main_training_data(input_dir):
	args = parse_args()
	datasets_name = args.datasets_name

	image_seq_dir = args.train_dataset_root_dir + 'training_datasets/'
	org_image_train_dir = args.image_dir + datasets_name + '/training/images/'
	org_swc_train_dir = args.image_dir + datasets_name + '/training/swc/'

	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'
	temp_image_dis_tif_dir = args.image_dir + datasets_name + '/temp/images_dis/'
	temp_skl_dis_tif_dir = args.image_dir + datasets_name + '/temp/centerlines_dis/'

	batch_size = args.input_dim
	BATCH_SHAPE = [batch_size[0],batch_size[1],batch_size[2]]
 
 
	image_name = input_dir.split("/")[-1].split(".")[0]

	print(image_name)

	if datasets_name == 'FMOST':
		img_dir = org_image_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '.adj.swc'

		fsize = os.path.getsize(swc_dir)
		total_fsize = 720000
		fsize_rate = fsize/total_fsize
		patch_num = round(args.N_patches * fsize_rate )

		resize_radio = 1.0
		swc_upsample_length = 99.0
		sample_gap = 1
		seq_len = 5
	else:
		pause


	img_new = open_tif(img_dir).astype(np.float32)


	data_image_dis_dir_tmp = temp_image_dis_tif_dir + image_name + '.img_dis.tif'
	if os.path.exists(data_image_dis_dir_tmp):
		img_dis_new = open_tif(data_image_dis_dir_tmp).astype(np.float32)
	else:
		org_img_o = copy.deepcopy(img_new)
		org_img_o[img_new > 500] = 500
		org_img_o = org_img_o/500*255
		org_img_f = np.ones_like(org_img_o)*255.0 - org_img_o
		org_img_f = org_img_f/255
		structure = generate_binary_structure(org_img_o.ndim, 1)
		img_dis_new = gwdt(org_img_f, structure)
		img_dis_new[img_dis_new>255]=255
		save_tif(img_dis_new, data_image_dis_dir_tmp, np.uint8)
 

	swc_tree_upsample = up_sample_swc_rescale(swc_dir, swc_upsample_length, resize_radio)
	data_swc_upsample_dir_tmp = temp_swc_centerline_dir + image_name + '.upsample.swc'
	swc_save(swc_tree_upsample, data_swc_upsample_dir_tmp)
			

	swc_tree_centerline = get_centerline_swc(data_swc_upsample_dir_tmp, 1.0)
	data_swc_centerline_dir_tmp = temp_swc_centerline_dir + image_name + '.centerline.swc'
	swc_save(swc_tree_centerline, data_swc_centerline_dir_tmp)
	swc_save_preorder(data_swc_centerline_dir_tmp, data_swc_centerline_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_centerline_dir_tmp) # read swc

	img_skl = save_swc2tif(data_swc_centerline_dir_tmp, [img_new.shape[0], img_new.shape[1], img_new.shape[2]])
	data_image_skl_dir_tmp = temp_centerline_dir + image_name + '_centerline_resize.tif'
	save_tif(img_skl, data_image_skl_dir_tmp, np.uint8)
	

	img_shape = img_new.shape
	img_skl_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1],img_shape[2]+4*BATCH_SHAPE[2]])
	img_skl_temp[2*BATCH_SHAPE[0]:2*BATCH_SHAPE[0] + img_shape[0], 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], 2*BATCH_SHAPE[2]:2*BATCH_SHAPE[2] + img_shape[2]] = copy.deepcopy(img_skl)

	data_skl_dis_dir_tmp = temp_skl_dis_tif_dir + image_name + '.skl_dis.tif'
	if os.path.exists(data_skl_dis_dir_tmp):
		img_skl_dis = open_tif(data_skl_dis_dir_tmp).astype(np.float32)
	else:
		img_skl_f = np.ones_like(img_skl_temp) - img_skl_temp//255
		img_skl_dis = morphology.distance_transform_edt(img_skl_f)
		img_skl_dis = img_skl_dis * 10
		img_skl_dis[img_skl_dis>255]=255
		save_tif(img_skl_dis, data_skl_dis_dir_tmp, np.uint8)

	swc_tree_exist = get_centerline_swc(data_swc_upsample_dir_tmp, 2.0)
	data_swc_exist_dir_tmp = temp_swc_centerline_dir + image_name + '.exist.swc'
	swc_save(swc_tree_exist, data_swc_exist_dir_tmp)
	swc_save_preorder(data_swc_exist_dir_tmp, data_swc_exist_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_exist_dir_tmp) # read swc

	label_new = save_swc2tif(data_swc_exist_dir_tmp, [img_new.shape[0], img_new.shape[1], img_new.shape[2]])
	data_image_exist_dir_tmp = temp_centerline_dir + image_name + '_exist_resize.tif'
	save_tif(label_new, data_image_exist_dir_tmp, np.uint8)


	swc_tree_centerline_matirx = read_swc_tree_matrix(data_swc_upsample_dir_tmp)
	swc_tree_centerline_matirx_radius = np.zeros([swc_tree_centerline_matirx.shape[0], 1])
	for i in range(swc_tree_centerline_matirx.shape[0]):
		swc_tree_centerline_matirx_radius[i][0] = swc_tree_centerline_matirx[i][5]

	swc_tree_centerline_matirx_direction = get_centerline_direction(swc_tree_centerline_matirx)
	swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector, swc_tree_centerline_matirx_v_vector = get_centerline_circle(swc_tree_centerline_matirx, swc_tree_centerline_matirx_direction)

 
	prepare_train_datasets(image_seq_dir, swc_tree_centerline_matirx, img_new, label_new, img_skl, img_dis_new, img_skl_dis, swc_tree_centerline_matirx_radius, swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector, swc_tree_centerline_matirx_v_vector, BATCH_SHAPE, image_name, patch_num, seq_len = seq_len, img_gap=sample_gap)
	
	time.sleep(1)


def main_test_data(input_dir):
	args = parse_args()
	datasets_name = args.datasets_name

	image_seq_dir = args.train_dataset_root_dir + 'test_datasets/'
	org_image_train_dir = args.image_dir + datasets_name + '/test/images/'
	org_swc_train_dir = args.image_dir + datasets_name + '/test/swc/'

	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'
	temp_image_dis_tif_dir = args.image_dir + datasets_name + '/temp/images_dis/'
	temp_skl_dis_tif_dir = args.image_dir + datasets_name + '/temp/centerlines_dis/'
	
	batch_size = args.input_dim
	BATCH_SHAPE = [batch_size[0],batch_size[1],batch_size[2]]
 
	image_name = input_dir.split("/")[-1].split(".")[0]

	print(image_name)

	if datasets_name == 'FMOST':
		img_dir = org_image_train_dir + image_name + '.tif'
		swc_dir = org_swc_train_dir + image_name + '.adj.swc'

		fsize = os.path.getsize(swc_dir)
		total_fsize = 453000
		fsize_rate = fsize/total_fsize
		patch_num = round(args.N_patches * fsize_rate // 10)

		resize_radio = 1.0
		swc_upsample_length = 99.0
		sample_gap = 1
		seq_len = 5
	else:
		pause


	
	img_new = open_tif(img_dir).astype(np.float32)

	data_image_dis_dir_tmp = temp_image_dis_tif_dir + image_name + '.img_dis.tif'
	if os.path.exists(data_image_dis_dir_tmp):
		img_dis_new = open_tif(data_image_dis_dir_tmp).astype(np.float32)
	else:
		org_img_o = copy.deepcopy(img_new)
		org_img_o[img_new > 500] = 500
		org_img_o = org_img_o/500*255
		org_img_f = np.ones_like(org_img_o)*255.0 - org_img_o
		org_img_f = org_img_f/255
		structure = generate_binary_structure(org_img_o.ndim, 1)
		img_dis_new = gwdt(org_img_f, structure)
		img_dis_new[img_dis_new>255]=255
		save_tif(img_dis_new, data_image_dis_dir_tmp, np.uint8)
 

	swc_tree_upsample = up_sample_swc_rescale(swc_dir, swc_upsample_length, resize_radio)
	data_swc_upsample_dir_tmp = temp_swc_centerline_dir + image_name + '.upsample.swc'
	swc_save(swc_tree_upsample, data_swc_upsample_dir_tmp)
			

	swc_tree_centerline = get_centerline_swc(data_swc_upsample_dir_tmp, 1.0)
	data_swc_centerline_dir_tmp = temp_swc_centerline_dir + image_name + '.centerline.swc'
	swc_save(swc_tree_centerline, data_swc_centerline_dir_tmp)
	swc_save_preorder(data_swc_centerline_dir_tmp, data_swc_centerline_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_centerline_dir_tmp) # read swc

	img_skl = save_swc2tif(data_swc_centerline_dir_tmp, [img_new.shape[0], img_new.shape[1], img_new.shape[2]])
	data_image_skl_dir_tmp = temp_centerline_dir + image_name + '_centerline_resize.tif'
	save_tif(img_skl, data_image_skl_dir_tmp, np.uint8)
	

	img_shape = img_new.shape
	img_skl_temp = np.zeros([img_shape[0]+4*BATCH_SHAPE[0],img_shape[1]+4*BATCH_SHAPE[1],img_shape[2]+4*BATCH_SHAPE[2]])
	img_skl_temp[2*BATCH_SHAPE[0]:2*BATCH_SHAPE[0] + img_shape[0], 2*BATCH_SHAPE[1]:2*BATCH_SHAPE[1] + img_shape[1], 2*BATCH_SHAPE[2]:2*BATCH_SHAPE[2] + img_shape[2]] = copy.deepcopy(img_skl)

	data_skl_dis_dir_tmp = temp_skl_dis_tif_dir + image_name + '.skl_dis.tif'
	if os.path.exists(data_skl_dis_dir_tmp):
		img_skl_dis = open_tif(data_skl_dis_dir_tmp).astype(np.float32)
	else:
		img_skl_f = np.ones_like(img_skl_temp) - img_skl_temp//255
		img_skl_dis = morphology.distance_transform_edt(img_skl_f)
		img_skl_dis = img_skl_dis * 10 
		img_skl_dis[img_skl_dis>255]=255
		save_tif(img_skl_dis, data_skl_dis_dir_tmp, np.uint8)

	swc_tree_exist = get_centerline_swc(data_swc_upsample_dir_tmp, 2.0)
	data_swc_exist_dir_tmp = temp_swc_centerline_dir + image_name + '.exist.swc'
	swc_save(swc_tree_exist, data_swc_exist_dir_tmp)
	swc_save_preorder(data_swc_exist_dir_tmp, data_swc_exist_dir_tmp) # pre-order
	swc_tree_centerline = read_swc_tree(data_swc_exist_dir_tmp) # read swc

	label_new = save_swc2tif(data_swc_exist_dir_tmp, [img_new.shape[0], img_new.shape[1], img_new.shape[2]])
	data_image_exist_dir_tmp = temp_centerline_dir + image_name + '_exist_resize.tif'
	save_tif(label_new, data_image_exist_dir_tmp, np.uint8)


	swc_tree_centerline_matirx = read_swc_tree_matrix(data_swc_upsample_dir_tmp)
	swc_tree_centerline_matirx_radius = np.zeros([swc_tree_centerline_matirx.shape[0], 1])
	for i in range(swc_tree_centerline_matirx.shape[0]):
		swc_tree_centerline_matirx_radius[i][0] = swc_tree_centerline_matirx[i][5]

	swc_tree_centerline_matirx_direction = get_centerline_direction(swc_tree_centerline_matirx)
	swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector, swc_tree_centerline_matirx_v_vector = get_centerline_circle(swc_tree_centerline_matirx, swc_tree_centerline_matirx_direction)

	prepare_train_datasets(image_seq_dir, swc_tree_centerline_matirx, img_new, label_new, img_skl, img_dis_new, img_skl_dis, swc_tree_centerline_matirx_radius, swc_tree_centerline_matirx_vector, swc_tree_centerline_matirx_u_vector, swc_tree_centerline_matirx_v_vector, BATCH_SHAPE, image_name, patch_num, seq_len = seq_len, img_gap=sample_gap)
	
	time.sleep(1)




if __name__ == '__main__':
	args = parse_args()
	
	datasets_name = args.datasets_name
	cpu_core_num = args.multi_cpu
	batch_size = args.input_dim
	patch_num = args.N_patches
 


	BATCH_SHAPE = [batch_size[0],batch_size[1],batch_size[2]]


	print("loading " + datasets_name + " datasets")
	

	org_image_train_dir = args.image_dir + datasets_name + '/training/images/'
	org_label_train_dir = args.image_dir + datasets_name + '/training/labels/'
	org_swc_train_dir = args.image_dir + datasets_name + '/training/swc/'

	org_image_test_dir = args.image_dir + datasets_name + '/test/images/'
	org_label_test_dir = args.image_dir + datasets_name + '/test/labels/'
	org_swc_test_dir = args.image_dir + datasets_name + '/test/swc/'

	temp_image_tif_dir = args.image_dir + datasets_name + '/temp/images/'
	temp_label_tif_dir = args.image_dir + datasets_name + '/temp/labels/'
	temp_swc_centerline_dir = args.image_dir + datasets_name + '/temp/swc_centerline/'
	temp_centerline_dir = args.image_dir + datasets_name + '/temp/centerline/'
	temp_image_dis_tif_dir = args.image_dir + datasets_name + '/temp/images_dis/'
	temp_skl_dis_tif_dir = args.image_dir + datasets_name + '/temp/centerlines_dis/'

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
	if not os.path.exists(temp_image_dis_tif_dir):
		os.makedirs(temp_image_dis_tif_dir)
	if not os.path.exists(temp_skl_dis_tif_dir):
		os.makedirs(temp_skl_dis_tif_dir)
		
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
	pool.map(main_training_data, org_label_list) 

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

	print('TOTAL %d images patches' % (total_test_data_num))



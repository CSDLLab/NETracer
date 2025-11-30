import torch
import torch.nn.init
import torch.utils.model_zoo as model_zoo
from torch import nn
import torch.nn.functional as F

class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, 1, padding=1, bias=True)
        self.bn = nn.BatchNorm3d(out_ch) 
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class up_conv_3d(nn.Module):
	"""
	Up Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(up_conv_3d, self).__init__()
		self.up = nn.Sequential(
			nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
			nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.up(x)
		return x

class down_conv_3d(nn.Module):
	"""
	Up Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(down_conv_3d, self).__init__()
		self.down = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
			nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.down(x)
		return x

class res_conv_block_3d(nn.Module):
	"""
	Res Convolution Block
	"""
	def __init__(self, out_ch):
		super(res_conv_block_3d, self).__init__()
		
		self.conv1 = nn.Sequential(
			nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm3d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm3d(out_ch))
		
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.conv1(x) + x)
		return out

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


class PE_Net_3D(nn.Module):
	"""
	Input: Original Image, Walked Image
	Output: Centerline, Point, Edge
	"""

	def __init__(self, in_ch=1, out_ch=1, freeze_net = False):
		super(PE_Net_3D, self).__init__()

		n1 = 32
		filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
		
		self.conv_input = ConvReLU(in_ch, filters[0])

		self.Conv1 = res_conv_block_3d(filters[0])
		self.Down1 = down_conv_3d(filters[0], filters[1])

		self.Conv2 = res_conv_block_3d(filters[1])
		self.Down2 = down_conv_3d(filters[1], filters[2])

		self.Conv3 = res_conv_block_3d(filters[2])
		self.Down3 = down_conv_3d(filters[2], filters[3])

		self.Conv4 = res_conv_block_3d(filters[3])
		self.Down4 = down_conv_3d(filters[3], filters[4])

		self.Conv5_1 = res_conv_block_3d(filters[4])

		self.Up5 = up_conv_3d(filters[4], filters[3])
		self.Up_conv5 = res_conv_block_3d(filters[3])

		self.Up4 = up_conv_3d(filters[3], filters[2])
		self.Up_conv4 = res_conv_block_3d(filters[2])

		self.Up3 = up_conv_3d(filters[2], filters[1])
		self.Up_conv3 = res_conv_block_3d(filters[1])

		self.Up2 = up_conv_3d(filters[1], filters[0])
		self.Up_conv2 = res_conv_block_3d(filters[0])
		
		self.centerline_block = res_conv_block_3d(filters[0])

		self.Conv_centerline = nn.Sequential(res_conv_block_3d(filters[0]),
                                           res_conv_block_3d(filters[0]))
		self.Conv_centerline_out = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)


		# walked path encoder
		self.walked_path_input = ConvReLU(in_ch, filters[0])
  
		# Point-Edge encoder
		self.point_edge_block = ConvReLU(filters[0]+filters[0], filters[0])
  
		self.Conv_edge = nn.Sequential(
			res_conv_block_3d(filters[0]),
			res_conv_block_3d(filters[0])
		)
		self.Conv_edge_out = nn.Sequential(
			res_conv_block_3d(filters[0]),
			nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
		)
		self.Conv_point = nn.Sequential(
			res_conv_block_3d(filters[0]),
			res_conv_block_3d(filters[0]),
			nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
		)
		
		self.active_Sigmoid = nn.Sigmoid()
		self.active_ReLU = nn.ReLU()


	def forward(self, input_image, walked_path):
		
		x = self.conv_input(input_image)
		e1 = self.Conv1(x)
		e1_d = self.Down1(e1)


		e2 = self.Conv2(e1_d)
		e2_d = self.Down2(e2)

		e3 = self.Conv3(e2_d)
		e3_d = self.Down3(e3)

		e4 = self.Conv4(e3_d)
		e4_d = self.Down4(e4)

		e5_2 = self.Conv5_1(e4_d)


		d5 = self.Up5(e5_2)
		d5 = torch.add(e4, d5)
		d5 = self.Up_conv5(d5)

		d4 = self.Up4(d5)
		d4 = torch.add(e3, d4)
		d4 = self.Up_conv4(d4)

		d3 = self.Up3(d4)
		d3 = torch.add(e2, d3)
		d3 = self.Up_conv3(d3)

		d2 = self.Up2(d3)
		d2 = torch.add(e1, d2)
		stage_fuse = self.Up_conv2(d2)

		centerline_fts = self.centerline_block(stage_fuse)
		
		# centerline encoder	
		d0_centerline = self.Conv_centerline(centerline_fts)
		d0_centerline_out = self.Conv_centerline_out(d0_centerline)
		centerline_out = self.active_Sigmoid(d0_centerline_out)
		
		# walked path encoder
		walked_path_f = self.walked_path_input(walked_path)
		stage_fuse = torch.cat([centerline_fts, walked_path_f], dim=1)

		# Point-Edge encoder
		point_edge_fts = self.point_edge_block(stage_fuse)
		# Edge block
		edge_fts = self.Conv_edge(point_edge_fts)
		edge_fts_out = self.Conv_edge_out(edge_fts)
		edge_out = self.active_Sigmoid(edge_fts_out)
  
		# Point block
		point_fts_input = torch.add(walked_path_f, edge_fts)
		point_fts = self.Conv_point(point_fts_input)
		point_out = self.active_Sigmoid(point_fts)
		
		return centerline_out, point_out, edge_out


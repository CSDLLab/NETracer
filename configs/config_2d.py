import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=80,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=int, default=0, help='use gpu only')
parser.add_argument("--local_rank", type=int, default=-1)

# parser.add_argument("--resize_radio", type=float, default=1.0) # road
# parser.add_argument("--r_resize", type=float, default=10)
# parser.add_argument("--r", type=float, default=3)

parser.add_argument("--resize_radio", type=float, default=2)  # drive 2 15
parser.add_argument("--r_resize", type=float, default=15)
parser.add_argument("--r", type=float, default=2)

# parser.add_argument("--resize_radio", type=float, default=1.5) # chasedb 1.5 15
# parser.add_argument("--r_resize", type=float, default=15)



# Datasets parameters DRIVE CHASEDB1 ROAD
parser.add_argument('--dataset_img_path', default = '/home/xxx/datasets/DRIVE/pointedgenet/training_data_v2/training_datasets/',help='Train datasets image root path')
parser.add_argument('--dataset_img_test_path', default = '/home/xxx/datasets/DRIVE/pointedgenet/training_data_v2/test_datasets/',help='Train datasets label root path')

parser.add_argument('--test_data_path', default = '/8T1/xxx/DRIVE/test/images_color/',help='Test datasets root path')
parser.add_argument('--test_data_mask_path', default = '/8T1/xxx/DRIVE/temp/mask/',help='Test datasets mask root path')

parser.add_argument('--predict_seed_path', default = '/8T1/xxxx/DRIVE/temp/frnet_seed_out/',help='Seed root path')
parser.add_argument('--predict_centerline_path', default = '/8T1/xxx/datasets/DRIVE/pointedgenet/pre/centerline/',help='Saved centerline result root path')
parser.add_argument('--predict_swc_path', default = '/8T1/xxx/datasets/DRIVE/pointedgenet/pre/swc/m1/',help='Saved swc result root path')

# DRIVE batch 256 model-id 12 
# # ROAD batch 128 model-id 1
parser.add_argument('--batch_size', type=int, default=256, help='batch size of trainset')
parser.add_argument('--valid_rate', type=float, default=0.1, help='')
parser.add_argument('--data_shape', type=list, default=[64,64], help='')

# data in/out and dataset
parser.add_argument('--model_save_dir', default='./model/model2d_',help='save path of trained model')
parser.add_argument('--log_save_dir', default='./log/log2d_',help='save path of trained log')

# train
parser.add_argument('--epochs', type=int, default=30, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=6, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=48)
parser.add_argument('--val_crop_max_size', type=int, default=96)
parser.add_argument('--hidden_layer_size', type=int, default=1)
parser.add_argument('--vector_bins', type=int, default=50)

# test
parser.add_argument('--use_amp', default=False, type=bool)
parser.add_argument('--train_or_test')
parser.add_argument('--to_restore', default=False, type=bool)

parser.add_argument('--test_patch_height', default=64)
parser.add_argument('--test_patch_width', default=64)
parser.add_argument('--stride_height', default=4)
parser.add_argument('--stride_width', default=4)

args = parser.parse_args()
import argparse
import template
import os
import scipy.io as scio

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='CMDT_2stg',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/CMDT/2stage/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='CMDT_2stg', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='Y',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi_PhiPhiT',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--parallel", type=bool, default=False)
parser.add_argument("--resume",type=bool,default=False)
parser.add_argument("--noisy",type=bool,default=False)

opt = parser.parse_args()
template.set_template(opt)

# dataset
file_abs_path=os.path.abspath(__file__)
abs_path_list=file_abs_path.split('/')
file_root_path='/'.join(abs_path_list[:-3])
train_data_path=file_root_path+'/datasets/cave_1024_28/'
test_data_path=file_root_path+'/datasets/TSA_simu_data/Truth/'
mask_path=file_root_path+'/datasets/TSA_simu_data/'
opt.data_path=train_data_path
opt.mask_path=mask_path
opt.test_path=test_data_path

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False

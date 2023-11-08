import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim
import torch_dct as dct
from einops import rearrange

def seed_everything(
    seed = 3407,
    deterministic = False, 
):
    """Set random seed.
    Args:
        seed (int): Seed to be used, default seed 3407, from the paper
        Torch. manual_seed (3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision[J]. arXiv preprint arXiv:2109.08203, 2021.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)
    return Phi_batch, Phi_s_batch

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training scences:', len(scene_list))
    num=len(scene_list)
    # num=2
    for i in range(num):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            img = img.astype(np.float32)
            imgs.append(img)
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    num=len(scene_list)
    # num=2
    test_data = np.zeros((num, 256, 256, 28))
    for i in range(num):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)['simulation_test']
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def shuffle_crop(train_data, batch_size, crop_size=256, argument=True):
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch

def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1],256,256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c,h,w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def init_mask(mask_path, mask_type, batch_size):
    mask3d_batch = generate_masks(mask_path, batch_size)
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask

def gen_meas_noisy(input_meas):
    
    mean=0
    sigma=0.1
    noise=torch.from_numpy(np.random.normal(mean,sigma,input_meas.shape)).cuda().float()
    out_meas=input_meas+noise
    out_meas[out_meas<0]=0
    # print("yes noise hdnet!")
    # print(out_meas.min(),out_meas.max())

    return out_meas

def init_meas(gt, mask, input_setting,noisy=False):
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    if noisy==True:
        input_meas=gen_meas_noisy(input_meas)
    return input_meas

# def checkpoint(model, epoch, model_path, logger):
#     model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
#     torch.save(model.state_dict(), model_out_path)
#     logger.info("Checkpoint saved to {}".format(model_out_path))

def checkpoint(model, epoch, model_path, logger,optimizer,scheduler):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save({'epoch':epoch,
                'model_dict':model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))
'''
# 1st weight,2nd square
def freq_loss(pred,gt):
    """
    pred shape:
    gt shape:
    """
    bs,nc,h,w=gt.shape
    kernel_size=8
    patch_h=int(h/kernel_size)
    patch_w=int(w/kernel_size)
    patch_num=int(patch_h*patch_w)
    pred_dct=dct.dct_2d(pred)
    gt_dct=dct.dct_2d(gt)
    pool=torch.nn.AvgPool2d(kernel_size=kernel_size,stride=kernel_size).cuda()
    tmp_weight=torch.abs(pool(pred_dct))
    one_matrix=torch.ones(tmp_weight.shape).cuda().float()
    tmp_weight=one_matrix/tmp_weight
    # rearrange(x_dct,'b c (hh m0) (ww m1) -> b (hh ww) c m0 m1',hh=kernel_num1,ww=kernel_num2,m0=kernel,m1=kernel)
    tmp_=tmp_weight.reshape(bs,nc,patch_num,1).repeat(1,1,1,kernel_size)
    tmp_=rearrange(tmp_,"b c (n m) k -> b c m (n k)",m=1)
    tmp_=tmp_.repeat(1,1,kernel_size,1)
    division=tmp_.chunk(patch_w,-1)
    alpha_weight=torch.cat(division[:],dim=-2)
    
    fl_tmp=torch.sum((alpha_weight*(pred_dct-gt_dct))**2)
    fl_tmp1=torch.sum((pred-gt)**2)
    n_sum=bs*nc*h*w
    floss1=torch.sqrt(fl_tmp1/n_sum)
    floss=torch.sqrt(fl_tmp/n_sum)
    
    return floss1,floss
'''

# 1st square,2nd weight
# def freq_loss(pred,gt):
#     """
#     pred shape:
#     gt shape:
#     """
#     bs,nc,h,w=gt.shape
#     kernel_size=8
#     patch_h=int(h/kernel_size)
#     patch_w=int(w/kernel_size)
#     patch_num=int(patch_h*patch_w)
#     pred_dct=dct.dct_2d(pred)
#     gt_dct=dct.dct_2d(gt)
#     pool=torch.nn.AvgPool2d(kernel_size=kernel_size,stride=kernel_size).cuda()
#     tmp_weight=torch.abs(pool(pred_dct))
#     tmp_weight[tmp_weight==0]=1.0
#     one_matrix=torch.ones(tmp_weight.shape).cuda().float()
#     tmp_weight=one_matrix/tmp_weight
#     # rearrange(x_dct,'b c (hh m0) (ww m1) -> b (hh ww) c m0 m1',hh=kernel_num1,ww=kernel_num2,m0=kernel,m1=kernel)
#     tmp_=tmp_weight.reshape(bs,nc,patch_num,1).repeat(1,1,1,kernel_size)
#     tmp_=rearrange(tmp_,"b c (n m) k -> b c m (n k)",m=1)
#     tmp_=tmp_.repeat(1,1,kernel_size,1)
#     division=tmp_.chunk(patch_w,-1)
#     alpha_weight=torch.cat(division[:],dim=-2)
    
#     fl_tmp=torch.sum(alpha_weight*(pred_dct-gt_dct)**2)
#     fl_tmp1=torch.sum((pred-gt)**2)
#     n_sum=bs*nc*h*w
#     floss1=torch.sqrt(fl_tmp1/n_sum)
#     floss=torch.sqrt(fl_tmp/n_sum)
    
#     return floss1,floss
# focal frequency loss
def freq_loss(pred,gt):
    """
    pred shape:
    gt shape:
    """
    bs,nc,h,w=gt.shape
    pred_dct=dct.dct_2d(pred)
    gt_dct=dct.dct_2d(gt)
    
    weight=torch.abs(pred_dct-gt_dct)
    fl_tmp=torch.sum(weight*((pred_dct-gt_dct)**2))
    # fl_tmp=torch.sum((pred-gt)**2)
    n_sum=bs*nc*h*w
    floss=torch.sqrt(fl_tmp/n_sum)
    # floss=torch.sqrt(fl_tmp/n_sum)
    
    return floss

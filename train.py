from architecture import *
from utils import *
seed_everything(
    seed = 3407,
    deterministic = True, 
)
# 3407  10.91
# 3207 11.34
import torch
import scipy.io as scio
import time
import os
import numpy as np
import torch_dct as dct
from torch.autograd import Variable
import datetime
from option import opt
import torch.nn.functional as F

# print("opt2 is ",opt)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,4,7'
print(torch.cuda.device_count())
# CUDA_LAUNCH_BLOCKING=1
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# init mask
mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
# mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, 2)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 10)
print(mask3d_batch_test.shape)
# mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 2)

# dataset
train_set = LoadTraining(opt.data_path)
test_data = LoadTest(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
# 3stage
# date_time="2023_04_25_23_24_18"
# 5stage
# date_time="2023_05_07_23_53_19"
# date_time="2023_07_03_23_10_09"
# 7stage
# date_time="2023_06_23_22_04_25"
# 9stage
# date_time="2023_08_29_19_17_10_v2"
result_path = opt.outf + date_time + '/result'
model_path = opt.outf + date_time + '/model'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path)
else:
    model = model_generator(opt.method, opt.pretrained_model_path)
    
# if opt.parallel: 
#     model = torch.nn.DataParallel(model) 

# model = model.cuda() 

# epoch_name=213
# pretrained_model_path=os.path.abspath(os.path.dirname(__file__))+"/exp/dauhst_2stg_patch_dct2d_attn_noshare/"+date_time+f"/model/model_epoch_{epoch_name}.pth"
# print(f"load model from epoch{epoch_name}")
# checkpoint_init=torch.load(pretrained_model_path)
# model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_init.items()},
#                               False)



start_epoch=0

# if opt.resume:
#     checkPoint=torch.load(opt.pretrained_model_path)
#     dictor=checkPoint['model_dict']
#     # for key in list(dict.keys()):
#     #     if key in ['module.denoiser.norm.weight','module.denoiser.norm.bias']:
#     #         del dict[key]
#     model.load_state_dict(dictor,strict=True) 
#     optimizer.load_state_dict(checkPoint['optimizer_dict'])
#     start_epoch = checkPoint['epoch']  
#     scheduler.load_state_dict(checkPoint['scheduler'])
    
# if opt.parallel: 
#     model = torch.nn.DataParallel(model) 

# model = model.cuda() 

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=[0.9, 0.999])
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
mse = torch.nn.MSELoss().cuda()

# if opt.resume:
    
#     optimizer.load_state_dict(checkPoint['optimizer_dict'])
#     start_epoch = checkPoint['epoch']  
#     scheduler.load_state_dict(checkPoint['scheduler'])

if opt.resume:
    print(opt.pretrained_model_path)
    checkPoint=torch.load(opt.pretrained_model_path)
    dictor=checkPoint['model_dict']
    # model.load_state_dict(dictor,strict=True)
    # for key in list(dict.keys()):
    #     if key in ['module.denoiser.norm.weight','module.denoiser.norm.bias']:
    #         del dict[key]

    weights_dict={}
    for k,v in checkPoint['model_dict'].items():
        new_k=k.replace('module.','') if 'module' in k else k
        weights_dict[new_k]=v
    model.load_state_dict(weights_dict,strict=False)

    
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkPoint['model_dict'].items()},strict=False)
    

if opt.parallel: 
    model = torch.nn.DataParallel(model) 
    print("para")

model = model.cuda() 
# optimizer=optimizer.cuda()
# scheduler=scheduler.cuda()
# optimizer.load_state_dict(checkPoint['optimizer_dict'])
# start_epoch = checkPoint['epoch']
# scheduler.load_state_dict(checkPoint['scheduler'])

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num // (opt.batch_size)))
    # if epoch>30:
    #     batch_num=2
    # batch_num=2
    for i in range(batch_num):
        gt_batch = shuffle_crop(train_set, opt.batch_size,argument=True)
        # gt_batch = shuffle_crop(train_set, 2)
        gt = Variable(gt_batch).cuda().float()
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting,opt.noisy).cuda()

        optimizer.zero_grad()
        # print(optimizer)

        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, diff_pred = model(input_meas, input_mask_train)
            loss = torch.sqrt(mse(model_out, gt))
            diff_gt = torch.mean(torch.abs(model_out.detach() - gt),dim=1, keepdim=True)  # [b,1,h,w]
            loss_sparsity = F.mse_loss(diff_gt, diff_pred)
            loss = loss + 2 * loss_sparsity
        else:
            model_out = model(input_meas, input_mask_train)
            loss = torch.sqrt(mse(model_out, gt))

        # if opt.method=='hdnet':
            # initF=torch.zeros((1,16,28,64,64),dtype=torch.complex32)
            # init_f=initF.cuda()
        # epoch_sdl_loss+=sdl_loss
        # fdl_loss = FDL_loss(model_out, gt)
        # epoch_fdl_loss+=fdl_loss
        # loss = sdl_loss + 0.7*fdl_loss
        # loss=sdl_loss
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f}, time: {:.2f}".
                format(epoch, epoch_loss / batch_num,(end - begin)))
    return 0

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting,opt.noisy)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, _ = model(input_meas, input_mask_test)
        else:
            model_out = model(input_meas, input_mask_test)

    # print(freq_comp_loss.shape)
    # print(mean_freq_comp_loss.shape)

    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("aver epoch=2,Learning rate:{}, batch_size:{}.argument=True\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(start_epoch+1, opt.max_epoch + 1+100):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 34.5:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(model, epoch, model_path, logger,optimizer,scheduler)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()



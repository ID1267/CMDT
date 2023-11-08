import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import numpy as np
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.ticker as ticker

path=os.path.abspath(os.path.dirname(__file__))

xmin=0
xmax=100
ymin=34
ymax=40
# fig=psnr.figure()
fig = plt.figure(figsize=(12,4))
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[1,1])
psnr=fig.add_subplot(spec[0,0])
px=psnr.grid(True,linestyle=(0,(8,10)),linewidth=0.6,alpha=1)
# psnr.grid(ls='--')
# psnr.grid()
psnr.set_xlim(xmin,xmax)
psnr.set_ylim(ymin,ymax)
# psnr.xlim(xmin,xmax)
# psnr.ylim(ymin,ymax)
psnr.set_xlabel('FlOPs(G)',fontsize=20)
psnr.set_ylabel('PSNR(dB)',fontsize=20)
# px.

# color
hdnet_color='#526D82'
mst_color='#76BA99'
cst_color='#80B3FF'
dau_color='#FFD966'
ours_color='#D80032'


# data
hdnet_x=[154.76]
hdnet_y=[34.97]
hdnet_params=np.array([2.37],dtype='float')
##
mst_x=[12.96,18.07,28.15]
mst_y=[34.26,34.94,35.18]
mst_params=np.array([0.93,1.5,2.0],dtype='float')
# mst_params=torch.from_numpy(mst_params).float()
# print(mst_params.shape)
cst_x=[11.67,16.91,27.81,40.10]
cst_y=[34.71,35.31,35.85,36.12]
cst_params=np.array([1.20,1.20,3.00,3.00],dtype='float')
##
dau_x=[18.44,27.17,44.61,79.50]
dau_y=[36.34,37.21,37.75,38.36]
dau_params=np.array([1.40,2.08,3.44,6.15],dtype='float')
##
ours_x=[21.39,31.56,51.90,92.59]
ours_y=[37.30,38.05,38.80,39.40]
ours_params=np.array([0.90]*4,dtype='float')

# draw
# hdnet
# psnr.scatter(hdnet_x,hdnet_y,s=hdnet_params*200,c=hdnet_color,alpha=0.8)
# psnr.plot(hdnet_x,hdnet_y,c=hdnet_color,alpha=0.8)
# psnr.text(hdnet_x[-1]-8,hdnet_y[-1]+0.8,s="HDNet",c=hdnet_color,fontsize=10)
# for i in range(hdnet_params.shape[0]):
#     psnr.text(hdnet_x[i]-2.5,hdnet_y[i]-0.04,s=str(hdnet_params[i])+'M',fontsize=5,verticalalignment="center")
# mst
psnr.scatter(mst_x,mst_y,s=mst_params*200,c=mst_color,alpha=0.8)
psnr.plot(mst_x,mst_y,c=mst_color,alpha=0.8)
psnr.text(mst_x[-1]+5,mst_y[-1]-0.07,s="MST(CVPR22)",fontsize=10,fontweight='bold')
for i in range(mst_params.shape[0]):
    psnr.text(mst_x[i]-2.5,mst_y[i]-0.04,s=str(mst_params[i]),fontsize=8,fontweight='bold',verticalalignment="center")
#  cst
psnr.scatter(cst_x,cst_y,s=cst_params*200,c=cst_color,alpha=0.8)
psnr.plot(cst_x,cst_y,c=cst_color,alpha=0.8)
psnr.text(cst_x[-1]+6,cst_y[-1]-0.08,s="CST(ECCV22)",fontsize=10,fontweight='bold')
for i in range(cst_params.shape[0]):
    psnr.text(cst_x[i]-2.4,cst_y[i]-0.04,s=str(cst_params[i]),fontsize=8,fontweight='bold',verticalalignment="center")
# dauhst
psnr.scatter(dau_x,dau_y,s=dau_params*200,c=dau_color,alpha=0.8)
psnr.plot(dau_x,dau_y,c=dau_color,alpha=0.8)
# psnr.text(dau_x[0]+4,dau_y[0]-0.2,s="DAUHST-2stg",fontsize=10)
# psnr.text(dau_x[1]-4,dau_y[1]-0.4,s="DAUHST-3stg",fontsize=10)
# psnr.text(dau_x[2]-6,dau_y[2]-0.6,s="DAUHST-5stg",fontsize=10)
psnr.text(dau_x[-1]-20,dau_y[-1]-0.8,s="DAUHST(NeurIPS22)",fontsize=10,fontweight='bold')
for i in range(dau_params.shape[0]):
    psnr.text(dau_x[i]-2.8,dau_y[i]-0.04,s=str(dau_params[i]),fontsize=8,fontweight='bold',verticalalignment="center")
# ours
psnr.scatter(ours_x,ours_y,s=ours_params*200,c=ours_color,alpha=0.8)
psnr.plot(ours_x,ours_y,c=ours_color,alpha=0.8)
psnr.text(ours_x[-1]-30,ours_y[-1],s="Ours",c=ours_color,fontsize=15,fontweight='bold')
for i in range(ours_params.shape[0]):
    psnr.text(ours_x[i]-2.3,ours_y[i]-0.03,s=str(ours_params[i]),fontsize=8,fontweight='bold',verticalalignment="center")

# ax=psnr.gca()
# ax.spines['bottom'].set_linewidth(0.1);###设置底部坐标轴的粗细
# ax.spines['left'].set_linewidth(0.1);####设置左边坐标轴的粗细
# ax.spines['right'].set_linewidth(0.1);###设置右边坐标轴的粗细
# ax.spines['top'].set_linewidth(0.1);####设置上部坐标轴的粗细

xmin2=0
xmax2=100
ymin2=0.93
ymax2=0.98
ssim=fig.add_subplot(spec[0,1])
px2=ssim.grid(True,linestyle=(0,(8,10)),linewidth=0.6,alpha=1)
# ssim.grid(ls='--')
# ssim.grid()
ssim.set_xlim(xmin2,xmax2)
ssim.set_ylim(ymin2,ymax2)
ssim.set_xlabel('FlOPs(G)',fontsize=20)
ssim.set_ylabel('SSIM',fontsize=20)

# color
hdnet_color='#526D82'
mst_color='#76BA99'
cst_color='#80B3FF'
dau_color='#FFD966'
ours_color='#D80032'


# data
hdnet_x2=[154.76]
hdnet_y2=[0.943]
hdnet_params2=np.array([2.37],dtype='float')
##
mst_x2=[12.96,18.07,28.15]
mst_y2=[0.935,0.943,0.948]
mst_params2=np.array([0.93,1.50,2.03],dtype='float')
# mst_params=torch.from_numpy(mst_params).float()
# print(mst_params.shape)
cst_x2=[11.67,16.91,27.81,40.10]
cst_y2=[0.940,0.947,0.954,0.957]
cst_params2=np.array([1.20,1.20,3.00,3.00],dtype='float')
##
dau_x2=[18.44,27.17,44.61,79.50]
dau_y2=[0.952,0.959,0.962,0.967]
dau_params2=np.array([1.40,2.08,3.44,6.15],dtype='float')
##
ours_x2=[21.39,31.56,51.90,92.59]
ours_y2=[0.963,0.967,0.972,0.976]
ours_params2=np.array([0.90]*4,dtype='float')

# draw
# hdnet
# ssim.scatter(hdnet_x,hdnet_y,s=hdnet_params*300,c=hdnet_color,alpha=0.8)
# ssim.plot(hdnet_x,hdnet_y,c=hdnet_color,alpha=0.8)
# ssim.text(hdnet_x[-1]-7,hdnet_y[-1]+0.004,s="HDNet",c=hdnet_color,fontsize=15)
# for i in range(hdnet_params.shape[0]):
#     ssim.text(hdnet_x[i]-3,hdnet_y[i],s=str(hdnet_params[i])+'M',fontsize=6,verticalalignment="center")
# mst
ssim.scatter(mst_x2,mst_y2,s=mst_params2*300,c=mst_color,alpha=0.8)
ssim.plot(mst_x2,mst_y2,c=mst_color,alpha=0.8)
ssim.text(mst_x2[-1]+5,mst_y2[-1]-0.001,s="MST(CVPR22)",fontsize=10,fontweight='bold')
for i in range(mst_params2.shape[0]):
    ssim.text(mst_x2[i]-2.7,mst_y2[i]-0.0002,s=str(mst_params2[i]),fontsize=8,fontweight='bold',verticalalignment="center")
#  cst
ssim.scatter(cst_x2,cst_y2,s=cst_params2*300,c=cst_color,alpha=0.8)
ssim.plot(cst_x2,cst_y2,c=cst_color,alpha=0.8)
ssim.text(cst_x2[-1]+6,cst_y2[-1]-0.001,s="CST(ECCV22)",fontsize=10,fontweight='bold')
for i in range(cst_params2.shape[0]):
    ssim.text(cst_x2[i]-2.4,cst_y2[i]-0.0002,s=str(cst_params2[i]),fontsize=8,fontweight='bold',verticalalignment="center")
# dauhst
ssim.scatter(dau_x2,dau_y2,s=dau_params2*300,c=dau_color,alpha=0.8)
ssim.plot(dau_x2,dau_y2,c=dau_color,alpha=0.8)
ssim.text(dau_x2[-1]-20,dau_y2[-1]-0.007,s="DAUHST(NeurIPS22)",fontsize=10,fontweight='bold')
for i in range(dau_params2.shape[0]):
    ssim.text(dau_x2[i]-2.8,dau_y2[i]-0.0002,s=str(dau_params2[i]),fontsize=8,fontweight='bold',verticalalignment="center")
# ours
ssim.scatter(ours_x2,ours_y2,s=ours_params2*300,c=ours_color,alpha=0.8)
ssim.plot(ours_x2,ours_y2,c=ours_color,alpha=0.8)
ssim.text(ours_x2[-1]-30,ours_y2[-1],s="Ours",c=ours_color,fontsize=15,fontweight='bold')
for i in range(ours_params2.shape[0]):
    ssim.text(ours_x2[i]-2.3,ours_y2[i]-0.0003,s=str(ours_params2[i]),fontsize=8,fontweight='bold',verticalalignment="center")

# savefig
# ssim.savefig(path+f"/ssim-flops.pdf",bbox_inches='tight')

fig.subplots_adjust(wspace=0.24)

# savefig
plt.savefig(path+f"/psnr-ssim-flops.png",bbox_inches='tight')

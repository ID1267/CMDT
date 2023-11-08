import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import numpy as np

path=os.path.abspath(os.path.dirname(__file__))

xmin=0
xmax=100
ymin=0.93
ymax=0.98
fig=plt.figure()
px=plt.grid(True,linestyle=(0,(8,10)),linewidth=0.6,alpha=1)
# plt.grid(ls='--')
# plt.grid()
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel('FlOPs(G)',fontsize=15)
plt.ylabel('SSIM',fontsize=15)

# color
hdnet_color='#526D82'
mst_color='#76BA99'
cst_color='#80B3FF'
dau_color='#FFD966'
ours_color='#D80032'


# data
hdnet_x=[154.76]
hdnet_y=[0.943]
hdnet_params=np.array([2.37],dtype='float')
##
mst_x=[12.96,18.07,28.15]
mst_y=[0.935,0.943,0.948]
mst_params=np.array([0.93,1.50,2.03],dtype='float')
# mst_params=torch.from_numpy(mst_params).float()
# print(mst_params.shape)
cst_x=[11.67,16.91,27.81,40.10]
cst_y=[0.940,0.947,0.954,0.957]
cst_params=np.array([1.20,1.20,3.00,3.00],dtype='float')
##
dau_x=[18.44,27.17,44.61,79.50]
dau_y=[0.952,0.959,0.962,0.967]
dau_params=np.array([1.40,2.08,3.44,6.15],dtype='float')
##
ours_x=[21.39,31.56,51.90,92.59]
ours_y=[0.963,0.967,0.972,0.976]
ours_params=np.array([0.90]*4,dtype='float')

# draw
# hdnet
# plt.scatter(hdnet_x,hdnet_y,s=hdnet_params*300,c=hdnet_color,alpha=0.8)
# plt.plot(hdnet_x,hdnet_y,c=hdnet_color,alpha=0.8)
# plt.text(hdnet_x[-1]-7,hdnet_y[-1]+0.004,s="HDNet",c=hdnet_color,fontsize=15)
# for i in range(hdnet_params.shape[0]):
#     plt.text(hdnet_x[i]-3,hdnet_y[i],s=str(hdnet_params[i])+'M',fontsize=6,verticalalignment="center")
# mst
plt.scatter(mst_x,mst_y,s=mst_params*300,c=mst_color,alpha=0.8)
plt.plot(mst_x,mst_y,c=mst_color,alpha=0.8)
plt.text(mst_x[-1]+4,mst_y[-1]-0.001,s="MST(CVPR22)",fontsize=10,fontweight='bold')
for i in range(mst_params.shape[0]):
    plt.text(mst_x[i]-2.5,mst_y[i],s=str(mst_params[i]),fontsize=6,verticalalignment="center")
#  cst
plt.scatter(cst_x,cst_y,s=cst_params*300,c=cst_color,alpha=0.8)
plt.plot(cst_x,cst_y,c=cst_color,alpha=0.8)
plt.text(cst_x[-1]+5,cst_y[-1]-0.001,s="CST(ECCV22)",fontsize=10,fontweight='bold')
for i in range(cst_params.shape[0]):
    plt.text(cst_x[i]-2.5,cst_y[i],s=str(cst_params[i]),fontsize=6,verticalalignment="center")
# dauhst
plt.scatter(dau_x,dau_y,s=dau_params*300,c=dau_color,alpha=0.8)
plt.plot(dau_x,dau_y,c=dau_color,alpha=0.8)
plt.text(dau_x[-1]-15,dau_y[-1]-0.006,s="DAUHST(NeurIPS22)",fontsize=10,fontweight='bold')
for i in range(dau_params.shape[0]):
    plt.text(dau_x[i]-2.5,dau_y[i],s=str(dau_params[i]),fontsize=6,verticalalignment="center")
# ours
plt.scatter(ours_x,ours_y,s=ours_params*300,c=ours_color,alpha=0.8)
plt.plot(ours_x,ours_y,c=ours_color,alpha=0.8)
plt.text(ours_x[-1]-30,ours_y[-1],s="Ours",c=ours_color,fontsize=15,fontweight='bold')
for i in range(ours_params.shape[0]):
    plt.text(ours_x[i]-2.5,ours_y[i],s=str(ours_params[i]),fontsize=6,verticalalignment="center")

# savefig
plt.savefig(path+f"/ssim-flops.pdf",bbox_inches='tight')

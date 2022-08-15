import numpy as onp
import jax.numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, value_and_grad
from jax.scipy.special import logsumexp
from jax.experimental import optimizers, stax
from jax.nn import selu,relu
from jax.experimental.stax import Dense, Relu, Sigmoid, Softplus, Selu, Tanh, Identity, FanOut, FanInConcat
from jax.numpy import tanh
from torch.utils.data import Dataset, DataLoader
from jax import random
import time
import pickle
import argparse
import math
from .utils import SDF_dataloader, plot_learning_curve
from .argument import args
from .nn_train import loss, batch_forward
import matplotlib.colors as mcolors

config = {'data_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF (1)/data/data_set/supervised_data.npy',
        'mode':'train',
        'train_loss_record_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF (1)/data/model/train_loss_record.npy',
        'test_loss_record_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF (1)/data/model/test_loss_record.npy',
        'if_test':False,
        'boundary_point_path':'/home/yawnlion/Desktop/PYproject/2D_deepSDF (1)/data/data_set/train_boundary_point.npy'}

def append_latent(latent_code,point):
    shape=point[args.num_dim].astype(int)
    latent=np.asarray(latent_code)[shape]
    in_array=np.concatenate((point[0:-1],latent))
    return in_array

batch_append_latent=vmap(append_latent,in_axes=(None,0))

matrix_append_latent=vmap(batch_append_latent,in_axes=(None,0))

mode = 'train'

file_read = open("/home/yawnlion/Desktop/PYproject/2D_deepSDF (1)/data/model/{}ed_params.txt".format(mode), "rb")
params = pickle.load(file_read)
nn = params[1]
latent_code = params[0]
boundary_point = onp.load('/home/yawnlion/Desktop/PYproject/2D_deepSDF (1)/data/data_set/{}_boundary_point.npy'.format(mode))


step=0.02
shape_1 = 3
shape_2 = 4
boundary_1 = boundary_point[shape_1]
x = onp.arange(-3,3,step)
y = onp.arange(-3,3,step)
X,Y = onp.meshgrid(x,y)
S_1 = np.ones(X.shape) * shape_1
S_2 = np.ones(X.shape) * shape_2
point_1 = np.stack([X,Y,S_1],axis=2)
in_array_1 = matrix_append_latent(latent_code, point_1)
point_2 = np.stack([X,Y,S_2],axis=2)
in_array_2 = matrix_append_latent(latent_code, point_2)
OUT_1 = batch_forward(nn, in_array_1)
OUT_1 = OUT_1.reshape(X.shape)
OUT_2 = batch_forward(nn, in_array_2)
OUT_2 = OUT_2.reshape(X.shape)

plt.figure(figsize=(5,5))
'''
contour = plt.contour(X,Y,OUT_1,[-0.5,0,0.5],colors='k')
x_b = boundary_1[:, 0]
y_b = boundary_1[:, 1]
plt.scatter(x_b, y_b, s = 1, c = 'r', marker = 'o')
plt.show()
'''
'''
OUT_1_shift = onp.ones(OUT_1.shape)
OUT_1 = onp.array(OUT_1)
print(OUT_2.shape)
OUT_2_seg = onp.array(OUT_2)
print(OUT_2.min())
print(OUT_2.max())
OUT_1_shift[40:,60:] = OUT_1[:-40,:-60]
OUT_2_seg[OUT_1_shift > 0.005] = 0
OUT_2_seg[OUT_1_shift < -0.005] = 0
#OUT_2_seg[OUT_1_shift != 0] = 0
print(OUT_2_seg.min())
print(OUT_2_seg.max())
norm = mcolors.DivergingNorm(vmin=OUT_2_seg.min(), vmax = OUT_2_seg.max(), vcenter=0)
plt.pcolor(x , y, OUT_2_seg, cmap='bwr',  norm=norm)
plt.colorbar()
contour1 = plt.contour(X,Y,OUT_2,[-0.3,0,0.6,1.5],colors='k')
plt.clabel(contour1,fontsize=10)
contour2 = plt.contour(X,Y,OUT_1_shift,[0,],colors='k',linewidth=0.5)
plt.title('SDF rendering of particle 1')
plt.xlabel('x(\mum)')
plt.ylabel('y(\mum)')
plt.axis('equal')
plt.show()
'''
OUT_1_shift = onp.ones(OUT_1.shape)
OUT_1 = onp.array(OUT_1)
print(OUT_2.shape)
OUT_2_seg = onp.array(OUT_2)
print(OUT_2.min())
print(OUT_2.max())
shape_2 = 3
x_b = boundary_point[shape_2][:, 0]
y_b = boundary_point[shape_2][:, 1]
plt.scatter(x_b, y_b, s = 10, c = 'black', marker = 'o',zorder=2)
OUT_1_shift[40:,60:] = OUT_1[:-40,:-60]
norm = mcolors.DivergingNorm(vmin=OUT_2.min(), vmax = OUT_2.max(), vcenter=0)
#plt.pcolor(x , y, OUT_2, cmap='RdBu',  norm=norm)
#plt.colorbar()
#contour1 = plt.contour(X,Y,OUT_2,[-0.5,0,0.5,1],cmap='RdBu',  norm=norm,zorder=1)
#plt.clabel(contour1,fontsize=10)

#contour2 = plt.contour(X,Y,OUT_1_shift,[0,],colors='k',linewidth=0.5)
#for i in range(len(boundary_point[shape_2])):
#    plt.plot([0., boundary_point[shape_1][i][0]], [0., boundary_point[shape_1][i][1]], 
#        linestyle='-',  linewidth=1, marker='o', markersize=5, color='black')
plt.title('SDF rendering of particle 1')
plt.xlabel('x(\mum)')
plt.ylabel('y(\mum)')
plt.axis('equal')
plt.show()
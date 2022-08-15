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
#from .nn_train import loss, batch_forward
import scipy.spatial as spt
import time


config = {'data_path':'data/data_set/supervised_data.npy',
        'mode':'train',
        'train_loss_record_path':'data/model/train_loss_record.npy',
        'test_loss_record_path':'data/model/test_loss_record.npy',
        'if_test':False,
        'boundary_point_path':'data/data_set/train_boundary_point.npy'}

batch_seeds = onp.load('data/data_set/{}_seeds.npy'.format(config['mode']), allow_pickle = True)

shape_ind = 1
'''
seeds = batch_seeds[shape_ind]
print(len(batch_seeds))
x_s = seeds[:, 0]
y_s = seeds[:, 1]
plt.scatter(x_s, y_s, s = 0.3, c = 'b', marker = 'v')
plt.show()

'''
#point = seeds
'''
hull = spt.ConvexHull(points=point, incremental=False)
print('闭合范围内的面积：', hull.area)

# 绘制散点图
plt.scatter(x=point[:, 0], y=point[:, 1], marker='*', c='b')
for sim in hull.simplices:
    # plt.plot([point[sim[0]][0], point[sim[1]][0]], [point[sim[0]][1], point[sim[1]][1]], 'g--')
    plt.plot(point[sim, 0], point[sim, 1], 'red')
plt.show()
'''
'''
kt = spt.KDTree(data=point, leafsize=10)  # 用于快速查找的KDTree类
ckt = spt.cKDTree(point)
gather =  kt.query(point[3], k = 3)
print(gather)
'''
#print(kt.data)
#print(batch_seeds.shape)
'''
T1 = time.perf_counter()
w_s = onp.arange((seeds.shape)[0]).astype(np.float64)
for i in range((seeds.shape)[0]):
        kt = spt.KDTree(data=point, leafsize=10)  # 用于快速查找的KDTree类
        ckt = spt.cKDTree(point)
        d, ind = kt.query(seeds[i], k = 3)
        w = (d[1] + d[2]) / 2.
        w_s[i] = w

T2 =time.perf_counter()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
print(w_s)
'''

index = len(batch_seeds)
error_all = []
batch_weights = []
for point in batch_seeds:
        error = []
        weights = []
        for i in range(len(point)):
                kt = spt.KDTree(data=point, leafsize=10)  # 用于快速查找的KDTree类
                ckt = spt.cKDTree(point)
                d, ind = kt.query(point[i], k = 3)
                #print(d[0])
                w = (d[1] + d[2]) / 2.
                weights.append(w)
                assert((point[ind[0]] - point[i]).all() == 0)
                if(np.dot(point[ind[2]] - point[i], -point[i] + point[ind[1]]) > 0):
                        weights.pop()
                        d, ind = kt.query(point[i], k = 4)
                        w = (d[1] + d[3]) / 2.
                        weights.append(w)
                        if(np.dot(point[ind[3]] - point[i], -point[i] + point[ind[1]]) > 0):
                                error.append(i)
                        #print(np.dot(point[ind[2]] - point[i], -point[i] + point[ind[1]]))
        error_all.append(error)
        batch_weights.append(weights)
        
onp.save('data/data_set/{}_batch_weights.npy'.format(config['mode']), batch_weights)
print(error_all)

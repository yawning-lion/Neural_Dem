import sys
#import PyGnuplot as gp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
#import matplotlib.pyplot as plt
import copy
import sys
#import PyGnuplot as gp

import numpy as onp
#import matplotlib.pyplot as plt
import copy
import jax.numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from jax import grad, jit, vmap, value_and_grad
from jax.example_libraries import optimizers, stax
from jax.nn import selu,relu
from jax.example_libraries.stax import Dense, Relu, Sigmoid, Softplus, Selu, Tanh, Identity, FanOut, FanInConcat
from jax.numpy import tanh
#from torch.utils.data import Dataset, DataLoader
from jax import random
import time
import pickle
import argparse
import math
from .utils import SDF_dataloader, plot_learning_curve
from .argument import args
from .nn_train import batch_forward



quads = onp.loadtxt("data/dat/surface_integral/sbi_case_3_quads.dat")
weights = onp.loadtxt("data/dat/surface_integral/sbi_case_3_weights.dat")
print(quads.shape)
print(weights.shape)
'''
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(quads[:,0], quads[:,1], quads[:,2])
plt.show()
'''
shape_ind = 1


file_read = open("data/model/{}ed_params.txt".format('train'), "rb")
params = pickle.load(file_read)
nn = params[1]
latent_code = params[0]

def torus2(x, y, z):
    value = 2*y*(y**2 - 3*x**2)*(1 - z**2) + (x**2 + y**2)**2 - (9*z**2 - 1)*(1 - z**2)
    return value

print(torus2(quads[:, 0]), quads[:, 1], quads[:, 2])


def single_shapeSDF(point, latent_code, nn):
    in_array = np.concatenate((point, latent_code))
    return batch_forward(nn, in_array)[0]

def batch_shapeSDF(point_batch, latent_code, nn):
    in_array = np.concatenate((point_batch, np.tile(latent_code, (point_batch.shape[0], 1))), 1)
    return batch_forward(nn, in_array)

single_grad_shapeSDF = grad(single_shapeSDF)

batch_grad_shapeSDF = vmap(single_grad_shapeSDF, in_axes=(0, None, None), out_axes=0)

print(single_grad_shapeSDF(np.array(quads[0]), latent_code[1], nn))
print(batch_grad_shapeSDF(np.array(quads), latent_code[2], nn))
print(quads)
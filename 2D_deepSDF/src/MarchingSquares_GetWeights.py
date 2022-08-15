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
from .nn_train import loss, batch_forward, single_forward, forward
from .Monte_Carlo_getseeds import run_dichotomy_loop

def generate_mesh(step = 0.05, bound = 3):
    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    X, Y = np.meshgrid(x, y)
    mesh = np.stack([X, Y], 2)
    return mesh
    
@jit
def get_neighbour(mesh, x_step = 0.05, y_step = 0.):
    X_neighbour = mesh[:, :, 0] + x_step
    Y_neighbour = mesh[:, :, 1] + y_step
    mesh_neighbour = np.stack([X_neighbour, Y_neighbour], 2)
    return mesh_neighbour
    

def single_check_line(point_A, point_B, nn, latent):
    line = np.stack([point_A, point_B], 0)
    latent_tiled = np.tile(latent, (2, 1))
    in_array = np.concatenate([line, latent_tiled], 1)
    sdf = batch_forward(nn, in_array)
    return np.where(np.sign(sdf[0]*sdf[1]) > 0, False, True)


line_check_line = vmap(single_check_line, in_axes = (0, 0, None, None), out_axes = 0)
matr_check_line = jit(vmap(line_check_line, in_axes = (0, 0, None, None), out_axes = 0))

@jit
def check_neighbour(mesh, x_step, y_step, nn, latent):
    mesh_neighbour = get_neighbour(mesh, x_step, y_step)
    check = matr_check_line(mesh, mesh_neighbour, nn, latent)
    check = np.squeeze(check, 2)
    return check, mesh_neighbour
    

def single_direction_select_line(mesh, x_step, y_step, nn, latent):
    check, mesh_neighbour = check_neighbour(mesh, x_step, y_step, nn, latent)
    print(check.shape)
    point_A_batch = mesh[check]
    point_B_batch = mesh_neighbour[check]
    single_direction_lines = np.stack([point_A_batch, point_B_batch], 1)
    return single_direction_lines
 
def point_to_index(line, step, bound):
    point = line[0]
    x = point[0]
    y = point[1]
    idx = (x + bound) / step
    idy = (y + bound) / step
    return round(idx), round(idy)
 
def get_spatial_map(pin, seeds, step, bound):
    length = round(2*bound / step)
    spatial_map = onp.zeros([length, length, 3])
    seeds_len = seeds.shape[0]
    for i in range(seeds_len):
        idx, idy = point_to_index(pin[i], step, bound)
        spatial_map[idx][idy][0] = 1
        spatial_map[idx][idy][1] = seeds[i][0]
        spatial_map[idx][idy][2] = seeds[i][1]
    return spatial_map

def MarchingCubes_get_pins(step, bound, nn, latent):
    mesh = generate_mesh(step, bound)
    x_direction_lines = single_direction_select_line(mesh, step, 0., nn, latent)
    y_direction_lines = single_direction_select_line(mesh, 0., step, nn, latent)
    pin_batch = np.concatenate([x_direction_lines, y_direction_lines], 0)
    return pin_batch
 
def MarchingCubes_getseeds_loop(step, bound, mode, num_seeds):
    file_read = open("/gpfs/share/home/1900011026/2D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    nn = params[1]
    latent_code = params[0]
    index = latent_code.shape[0]
    batch_seeds = []
    for i in range(1):
        pin_batch = MarchingCubes_get_pins(step, bound, nn, latent_code[i])
        all_seeds = run_dichotomy_loop(pin_batch, latent_code[i], nn, 10)
        all_num = all_seeds.shape[0]
        selector = onp.random.choice(np.arange(all_num), size = num_seeds, replace=False)
        seeds = all_seeds[selector]
        batch_seeds.append(seeds)
        print(f'{mode} shape {i} done')
    batch_seeds = np.asarray(batch_seeds)
    onp.save('/gpfs/share/home/1900011026/2D_deepSDF/data/data_set/{}_seeds.npy'.format(mode), batch_seeds)
    return batch_seeds  
    
def x_get_weights(x_map, y_map, x_pin, x_seeds, step, bound):
    seeds_len = seeds.shape[0]
    weighted_seeds = onp.zeros([seeds_len, 3])
    for i in range(seeds_len):
        idx, idy = point_to_index(x_pin[i], step, bound)
        potential_local_seeds = onp.stack(
            x_map[idx][idy + 1],
            x_map[idx][idy - 1],
            y_map[idx][idy],
            y_map[idx][idy - 1],
            y_map[idx + 1][idy],
            y_map[idx + 1][idy - 1])
        for j in range(6):
            weight += (potential_local_seeds[0] * np.sqrt((potential_local_seeds[1] - x_seeds[0])**2 + (potential_local_seeds[2] - x_seeds[1])**2))/2
        weighted_seeds[i][0] = x_seeds[0]
        weighted_seeds[i][1] = x_seeds[1]
        weighted_seeds[i][2] = weight
    return weighted_seeds

def get_weights(x_map, y_map, x_pin, x_seeds, y_seeds, step, bound):
    x_weighted_seeds = x_get_weights(x_map, y_map, x_pin, x_seeds, step, bound);
    y_weighted_seeds = x_get_weights(y_map.tranpose((0,2,1)), x_map.tranpose((0,2,1))), y_pin, y_seeds, step, bound);
    weighted_seeds = np.concatenate((x_weighted_seeds, y_weighted_seeds),0);
    return weighted_seeds
    
    



if __name__ == '__main__':
    '''
    mode = 'train'
    file_read = open("/gpfs/share/home/1900011026/2D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    nn = params[1]
    latent_code = params[0]

    index = latent_code.shape[0]
    batch_seeds = []
    step = 0.2
    bound = 3
    mesh = generate_mesh(step, bound)
    i = 1
    latent = latent_code[i]
    x_direction_lines = single_direction_select_line(mesh, step, 0., nn, latent)
    y_direction_lines = single_direction_select_line(mesh, 0., step, nn, latent)
    x_seeds = run_dichotomy_loop(x_direction_lines, latent_code[i], nn, 10)
    y_seeds = run_dichotomy_loop(y_direction_lines, latent_code[i], nn, 10)
    x_map = get_spatial_map(x_direction_lines, x_seeds, step, bound)
    y_map = get_spatial_map(y_direction_lines, y_seeds, step, bound)
    '''
    #onp.save('/gpfs/share/home/1900011026/2D_deepSDF/data/data_set/{}_seeds.npy'.format(mode), batch_seeds)
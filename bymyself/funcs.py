import numba as nb
import numpy as np
from numba import jit, njit
import h5py
import os

# 周期性模拟盒子中的最短距离
@njit
def dist(x,y,boxSize):
    diff = x-y
    diff[np.where(diff>boxSize/2)[0]] -= boxSize
    diff[np.where(diff<=-boxSize/2)[0]] += boxSize
    r=np.linalg.norm(diff)
    return r

# 一次性计算一个点 x 到多个三维点 y 的周期性距离
@njit(parallel = True)
def norm_axis1(arr):
    assert len(arr.shape) == 2
    norms = np.empty(arr.shape[0], dtype = arr.dtype)
    for i in nb.prange(arr.shape[0]):
        norms[i] = np.sqrt(np.sum(arr[i,:]*arr[i,:]))
    return norms

@njit(parallel = True)
def dist_vector_nb(x,y,boxSize):
    assert x.shape == (3,), 'x must be of shape (3,)'
    assert y.shape[1] == 3, 'y must be an array of 3-vectors'
    r = np.empty(y.shape[0], dtype=np.float64)

    for i in nb.prange(y.shape[0]):
        dx = x[0] - y[i, 0]
        dy = x[1] - y[i, 1]
        dz = x[2] - y[i, 2]

        if dx > boxSize / 2:
            dx -= boxSize
        elif dx <= -boxSize / 2:
            dx += boxSize

        if dy > boxSize / 2:
            dy -= boxSize
        elif dy <= -boxSize / 2:
            dy += boxSize

        if dz > boxSize / 2:
            dz -= boxSize
        elif dz <= -boxSize / 2:
            dz += boxSize
        r[i] = np.sqrt(dx * dx + dy * dy + dz * dz)

    return r

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
@njit
def norm_axis1(arr):
    assert len(arr.shape) == 2
    norms = np.empty(arr.shape[0], dtype = arr.dtype)
    for i in nb.prange(arr.shape[0]):
        norms[i] = np.sqrt(np.sum(arr[i,:]*arr[i,:]))
    return norms

@njit
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

# 按照 xVal 分箱，然后计算每个分箱中 yVal 的：中位数；下百分位数；上百分位数
def binData_med(xVal, yVal, numBins=4, lower = 16, upper = 84):
    minVal = np.min(xVal)
    maxVal = np.max(xVal)

    binWidth = (maxVal - minVal) / numBins

    xMed = np.full(numBins, np.nan)
    yMed = np.full(numBins, np.nan)
    ylow = np.full(numBins, np.nan)
    yup = np.full(numBins, np.nan)

    for j in range(numBins):
        relInd = np.where( (xVal >= minVal + j*binWidth) & (xVal < minVal + (j+1)*binWidth) )[0]
        if(relInd.size>0):
            xMed[j] = np.nanmedian(xVal[relInd])
            yMed[j] = np.nanmedian(yVal[relInd])
            ylow[j] = np.nanpercentile(yVal[relInd],lower)
            yup[j] = np.nanpercentile(yVal[relInd],upper)

    return xMed, yMed, ylow, yup

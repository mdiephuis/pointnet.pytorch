import numpy as np
import torch

def AddGaussianNoise(pointcloud, sigma = 0.001):
    ''' The function adds zero-mean Gaussian Noise with variance = sigma
    to all the points of a point Cloud'''
    N,D = pointcloud.shape
    mu, sigma = 0, sigma  # mean and standard deviation
    noise = torch.randn(N,D) * sigma #np.random.normal(mu, sigma, (N,D))
    noisypointcloud = pointcloud + noise
    return noisypointcloud
import numpy as np
import torch

def AddGaussianNoise(pointcloud, sigma = 0.001, percent_points = 0.5):
    ''' The function adds zero-mean Gaussian Noise with variance = sigma
    to all the points of a point Cloud'''
    N,D = pointcloud.shape
    mu, sigma = 0, sigma  # mean and standard deviation
    noise = np.random.rand(N,D) * sigma #np.random.normal(mu, sigma, (N,D))

    #turn off randomly some part of noise, to apply it only to a part of points
    nums = np.random.choice([0, 1], size=(N,1), p=[(1-percent_points), percent_points])
    noisypointcloud = pointcloud + (nums*noise)
    return noisypointcloud
import numpy as np


def add_gaussian_noise(pointcloud, sigma=0.1, percent_points=0.2):
    ''' The function adds zero-mean Gaussian Noise with variance = sigma
    to random points of a point Cloud, with a given ratior percent_points'''
    N, D = pointcloud.shape
    noise = np.random.randn(N, D)  # np.random.normal(mu, sigma, (N,D))
    noise = normalize(noise) * sigma

    # turn off randomly some part of noise, to apply it only to a part of points
    # nums = np.random.choice([0, 1], size=(
    #     N, 1), p=[(1 - percent_points), percent_points])
    # # multiply the point cloud by noise vector
    noisypointcloud = pointcloud + noise
    # clamp [-1, 1]

    return noisypointcloud


def normalize(x):
    min_x = np.min(x)
    max_x = np.max(x)
    return -1 + (2.0 / (max_x - min_x) * (x - min_x))

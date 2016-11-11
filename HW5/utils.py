import math

import numpy as np


def multivariate_gaussian(x, mean, var):
    d = len(var)
    p = 1/ ( math.pow(2 * math.pi, 1/d) * math.pow(np.linalg.norm(var),1/2) )
    p *= math.exp( -0.5*( np.dot(np.dot(np.transpose(np.subtract(x, mean)), np.linalg.inv(var)), np.subtract(x, mean)))  )
    return p


def get_diff_norm(v1, v2, order=2):
    return np.linalg.norm(np.subtract(v1, v2), ord=order)


def generate_rgb_colors(num):
    return get_n_points_in_range(num, 3, (0,0,0), (1,1,1))


def get_n_integer_points_in_range(num_points, d, minvals, maxvals):
    point_generator = get_random_integer_point_in_range_generator(d, minvals, maxvals)
    points = [next(point_generator) for x in range(0, num_points)]
    return points


def get_n_points_in_range(num_points, d, minvals, maxvals):
    point_generator = get_random_point_in_range_generator(d, minvals, maxvals)
    points = [next(point_generator) for x in range(0, num_points)]
    return points


def get_random_point_in_range_generator(d, minvals, maxvals):
    while True:
        yield get_random_point_in_range(d, minvals, maxvals)


def get_random_integer_point_in_range_generator(d, minvals, maxvals):
    while True:
        yield get_random_integer_point_in_range(d, minvals, maxvals)


def get_random_integer_point_in_range(d, minvals, maxvals):
    point = []
    for f in range(0, d):
        point.append(np.random.randint(minvals[f], maxvals[f]))
    return point

def get_random_point_in_range(d, minvals, maxvals):
    point = []
    for f in range(0, d):
        point.append(np.random.uniform(minvals[f], maxvals[f]))
    return point

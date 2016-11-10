import numpy as np


def generate_rgb_colors(num):
    return get_n_points_in_range(num, 3, (0,0,0), (1,1,1))


def get_n_points_in_range(num_points, d, minvals, maxvals):
    point_generator = get_random_point_in_range_generator(d, minvals, maxvals)
    points = [next(point_generator) for x in range(0, num_points)]
    return points


def get_random_point_in_range_generator(d, minvals, maxvals):
    while True:
        yield get_random_point_in_range(d, minvals, maxvals)


def get_random_point_in_range(d, minvals, maxvals):
    point = []
    for f in range(0, d):
        point.append(np.random.uniform(minvals[f], maxvals[f]))
    return point

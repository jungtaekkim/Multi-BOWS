import numpy as np


path_results = '../results'

time_budgets_target = [
    ['threelayer', 5000.0],
    ['matched', 20000.0],
    ['unmatched', 80000.0],
    ['automatic', 80000.0],
]

range_X_mingxuan = np.array([
    [3.0, 20.0], # thickness_silver, 0
    [5.0, 100.0], # thickness_up, 1
    [5.0, 100.0], # thickness_down, 2
    [10.0, 200.0], # radius_up_bottom, 3
    [10.0, 200.0], # radius_down_top, 4
    [50.0, 800.0], # height_up, 5
    [50.0, 800.0], # height_down, 6
    [20.0, 400.0], # grid_size, 7
    [0.05, 0.999], # ratio radius_up_top/radius_up_bottom, 8
    [0.05, 0.999], # ratio radius_down_bottom/radius_down_top, 9
])

range_X = np.array([
    [2.5001, 20.4999], # thickness_silver, 0
    [4.5001, 100.4999], # thickness_up, 1
    [4.5001, 100.4999], # thickness_down, 2
    [9.5001, 50.4999], # radius_up_bottom, 3
    [9.5001, 50.4999], # radius_down_top, 4
    [49.5001, 400.4999], # height_up, 5
    [49.5001, 400.4999], # height_down, 6
    [20.0, 100.0], # grid_size_up, 7
    [0.0, 0.9999], # ratio radius_up_top/radius_up_bottom, 8
    [0.0, 0.9999], # ratio radius_down_bottom/radius_down_top, 9
    [0.5001, 10.4999], # the number of upper cones, 10
    [0.5001, 10.4999], # the number of lower cones, 11
])

range_X_to_verify = np.array([
    [3, 20],
    [5, 100],
    [5, 100],
    [1, 99],
    [10, 100],
    [10, 100],
    [1, 99],
    [50, 400],
    [50, 400],
    [20, 400],
    [20, 400],
    [1, 10],
    [1, 10],
])

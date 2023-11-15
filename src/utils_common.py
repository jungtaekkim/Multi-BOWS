import os
from datetime import datetime
import numpy as np

import constants


def print_separator():
    print('=============================================', flush=True)

def print_info(
    thickness_silver,
    thickness_up,
    thickness_down,
    radius_up_top,
    radius_up_bottom,
    radius_down_top,
    radius_down_bottom,
    height_up,
    height_down,
    grid_size,
    grid_size_up,
    grid_size_down,
    delta_x,
    delta_y,
    num_cones_up,
    num_cones_down,
    min_meshstep,
):
    print_separator()
    print(f'All values are in nanometers.')
    print(f'thickness: silver {thickness_silver:.4f} up {thickness_up:.4f} down {thickness_down:.4f}')
    print(f'radius_up: top {radius_up_top:.4f} < bottom {radius_up_bottom:.4f}')
    print(f'radius_down: top {radius_down_top:.4f} > bottom {radius_down_bottom:.4f}')
    print(f'height: up {height_up:.4f} down {height_down:.4f}')
    print(f'grid_size: {grid_size:.4f} up {grid_size_up:.4f} down {grid_size_down:.4f}')
    print(f'delta: x {delta_x:.4f} y {delta_y:.4f}')
    print(f'num_cones: up {num_cones_up} down {num_cones_down}')
    print(f'min_meshstep {min_meshstep}')
    print_separator()
    print(f'')

def get_str_now():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

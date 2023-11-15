import numpy as np

import exp_nanocones


# 0: thickness_silver
# 1: thickness_up
# 2: thickness_down
# 3: radius_up_top
# 4: radius_up_bottom
# 5: radius_down_top
# 6: radius_down_bottom
# 7: height_up
# 8: height_down
# 9: grid_size
# 10: delta_x
# 11: delta_y
# 12: num_cones_up
# 13: num_cones_down
# 14: min_meshstep


def fun_target_threelayer(bx, min_meshstep):
    # 0: thickness_silver
    # 1: thickness_up
    # 2: thickness_down
    assert bx.shape[0] == 3

    return exp_nanocones.run_simulation(
        bx[0],
        bx[1],
        bx[2],
        None,
        None,
        None,
        None,
        None,
        None,
        20.0,
        None,
        None,
        None,
        None,
        min_meshstep,
    )

def fun_target_matched(bx, min_meshstep):
    # 0: thickness_silver
    # 1: thickness_up
    # 2: thickness_down
    # 3: radius_up_bottom
    # 4: radius_down_top
    # 5: height_up
    # 6: height_down
    # 7: grid_size
    # 8: RATIO radius_up_top / radius_up_bottom
    # 9: RATIO radius_down_bottom / radius_down_top
    assert bx.shape[0] == 10

    fun_round = np.round

    delta_x = delta_y = 0.0
    num_cones = 1

    radius_up_top = fun_round(bx[3] * bx[8])
    radius_up_top = np.maximum(1, radius_up_top)
    radius_up_top = np.minimum(radius_up_top, fun_round(bx[3]) - 1)
    radius_down_bottom = fun_round(bx[4] * bx[9])
    radius_down_bottom = np.maximum(1, radius_down_bottom)
    radius_down_bottom = np.minimum(radius_down_bottom, fun_round(bx[4]) - 1)

    grid_size = np.maximum(2 * np.maximum(fun_round(bx[3]), fun_round(bx[4])), bx[7])

    return exp_nanocones.run_simulation(
        bx[0],
        bx[1],
        bx[2],
        radius_up_top,
        bx[3],
        bx[4],
        radius_down_bottom,
        bx[5],
        bx[6],
        grid_size,
        delta_x,
        delta_y,
        num_cones,
        num_cones,
        min_meshstep,
    )

def fun_target_unmatched(bx, min_meshstep):
    # 0: thickness_silver
    # 1: thickness_up
    # 2: thickness_down
    # 3: radius_up_bottom
    # 4: radius_down_top
    # 5: height_up
    # 6: height_down
    # 7: grid_size_up
    # 8: RATIO radius_up_top / radius_up_bottom
    # 9: RATIO radius_down_bottom / radius_down_top
    # 10: the number of upper cones
    # 11: the number of lower cones
    assert bx.shape[0] == 12

    fun_round = np.round

    delta_x = delta_y = 0.0

    radius_up_bottom = fun_round(bx[3])

    grid_size_up = np.maximum(2 * radius_up_bottom, bx[7])
    grid_size = grid_size_up * fun_round(bx[10])

    grid_size_down = grid_size / fun_round(bx[11])

    if grid_size_down < 20.0:
        num_cones_down = grid_size // 20
    elif grid_size_down > 400.0:
        num_cones_down = np.ceil(grid_size / 400)
    else:
        num_cones_down = fun_round(bx[11])

    grid_size_down = grid_size / num_cones_down
    radius_down_top = np.minimum(fun_round(bx[4]), np.floor(grid_size_down / 2))

    radius_up_top = fun_round(radius_up_bottom * bx[8])
    radius_up_top = np.maximum(1, radius_up_top)
    radius_up_top = np.minimum(radius_up_top, radius_up_bottom - 1)
    radius_down_bottom = fun_round(radius_down_top * bx[9])
    radius_down_bottom = np.maximum(1, radius_down_bottom)
    radius_down_bottom = np.minimum(radius_down_bottom, radius_down_top - 1)

    return exp_nanocones.run_simulation(
        bx[0],
        bx[1],
        bx[2],
        radius_up_top,
        radius_up_bottom,
        radius_down_top,
        radius_down_bottom,
        bx[5],
        bx[6],
        grid_size,
        delta_x,
        delta_y,
        int(fun_round(bx[10])),
        int(num_cones_down),
        min_meshstep,
    )

def fun_target_automatic(bx, min_meshstep):
    # 0: thickness_silver
    # 1: thickness_up
    # 2: thickness_down
    # 3: radius_up_bottom
    # 4: radius_down_top
    # 5: height_up
    # 6: height_down
    # 7: grid_size_up
    # 8: RATIO radius_up_top / radius_up_bottom
    # 9: RATIO radius_down_bottom / radius_down_top
    # 10: the number of upper cones
    # 11: the number of lower cones
    # 12: structure selection
    assert bx.shape[0] == 13

    fun_round = np.round

    structure_selection = fun_round(bx[12])

    if structure_selection == 0:
        return fun_target_threelayer(bx[:3], min_meshstep)
    elif structure_selection == 1:
        delta_x = delta_y = 0.0
        num_cones = 1

        radius_up_top = fun_round(bx[3] * bx[8])
        radius_up_top = np.maximum(1, radius_up_top)
        radius_up_top = np.minimum(radius_up_top, fun_round(bx[3]) - 1)

        grid_size = np.maximum(2 * fun_round(bx[3]), bx[7])

        return exp_nanocones.run_simulation(
            bx[0],
            bx[1],
            bx[2],
            radius_up_top,
            bx[3],
            None,
            None,
            bx[5],
            None,
            grid_size,
            delta_x,
            delta_y,
            num_cones,
            None,
            min_meshstep,
        )
    elif structure_selection == 2:
        delta_x = delta_y = 0.0
        num_cones = 1

        radius_down_bottom = fun_round(bx[4] * bx[9])
        radius_down_bottom = np.maximum(1, radius_down_bottom)
        radius_down_bottom = np.minimum(radius_down_bottom, fun_round(bx[4]) - 1)

        grid_size = np.maximum(2 * fun_round(bx[4]), bx[7])

        return exp_nanocones.run_simulation(
            bx[0],
            bx[1],
            bx[2],
            None,
            None,
            bx[4],
            radius_down_bottom,
            None,
            bx[6],
            grid_size,
            delta_x,
            delta_y,
            None,
            num_cones,
            min_meshstep,
        )

    elif structure_selection == 3:
        return fun_target_matched(bx[:10], min_meshstep)
    elif structure_selection == 4:
        return fun_target_unmatched(bx[:12], min_meshstep)
    else:
        raise ValueError


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    debug = args.debug

    list_min_meshstep = [1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19, 20]
    list_transparency = []

    for min_meshstep in list_min_meshstep:
        list_transparency.append(fun_target_threelayer(np.array([10.0, 40.0, 10.0]), min_meshstep)[0])

    print(list_transparency)

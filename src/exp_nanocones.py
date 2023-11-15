import lumapi

import time
import os
import numpy as np

import shielding_effectiveness
import constants
import utils_common


NANO = 1e-9

AUTOSHUTOFF = 1e-6
MESH_ACCURACY = 2

dz_silver = 1.0 * NANO
dz_up = dz_down = 1.0 * NANO


def compute_cone(radius_bottom, radius_top, height):
    theta = np.arctan((radius_bottom - radius_top) / height) # half angle of cone tip
    ht = radius_top / np.tan(theta) # clipped length of cone tip

    return theta, ht

def set_plate(
    fdtd, str_up_or_down,
    z_min, z_max,
    width_simulation
):
    fdtd.addrect()
    fdtd.set("name", f"highindex_{str_up_or_down}")
    fdtd.set("x", 0.0)
    fdtd.set("y", 0.0)
    fdtd.set("z min", z_min)
    fdtd.set("z max", z_max)
    fdtd.set("x span", width_simulation)
    fdtd.set("y span", width_simulation)
    fdtd.set("material", "TiO2")

def set_cones(
    fdtd, str_up_or_down,
    pitch, a,
    radius_bottom, radius_top,
    height, z,
    delta_x=None, delta_y=None,
):
    num_cones = int(np.round(pitch / a)) + 1
    theta, ht = compute_cone(radius_bottom, radius_top, height)
    print(f'num_cones_{str_up_or_down} {num_cones} x {num_cones}', flush=True)

    offset_x = -pitch / 2.0
    offset_y = -pitch / 2.0
    z_cone = z - height / 2.0

    if str_up_or_down == "up":
        assert delta_x is not None and delta_y is not None
        assert isinstance(delta_x, float)
        assert isinstance(delta_y, float)

        offset_x += delta_x
        offset_y += delta_y
        z_cone += height

    fdtd.addstructuregroup()
    fdtd.set("name", f"cones_{str_up_or_down}")

    fdtd.addstructuregroup()
    fdtd.set("name", f"cones_{str_up_or_down}_0")

    for ind in range(0, num_cones):
        fdtd.addcustom()

        fdtd.set("name", f"cone_{str_up_or_down}_{ind}")
        fdtd.set("x", offset_x + ind * a)
        fdtd.set("y", offset_y)
        fdtd.set("z", z_cone)

        fdtd.set("first axis", "y")
        fdtd.set("rotation 1", 90)
        fdtd.set("x span", height)
        fdtd.set("y span", 2 * np.maximum(radius_bottom, radius_top))
        fdtd.set("z span", 2 * np.maximum(radius_bottom, radius_top))
        fdtd.set("create 3D object by", "revolution")

        eqn = str(radius_top / ht) + "* (x + " + str((height / 2.0 + ht) * 1e6) + ")"
        fdtd.set("equation 1", eqn)
        fdtd.set("material", "TiO2")

        fdtd.addtogroup(f"cones_{str_up_or_down}_0")

    fdtd.select(f"cones_{str_up_or_down}_0")
    fdtd.addtogroup(f"cones_{str_up_or_down}")

    for ind in range(1, num_cones):
        fdtd.select(f"cones_{str_up_or_down}::cones_{str_up_or_down}_0")
        fdtd.copy(0.0, ind * a)
        fdtd.set("name", f"cones_{str_up_or_down}_{ind}")

def run_simulation(
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
    delta_x,
    delta_y,
    num_cones_up,
    num_cones_down,
    min_meshstep,
    debug=False,
):
    # thickness_silver
    # thickness_up
    # thickness_down
    # radius_up_top
    # radius_up_bottom
    # radius_down_top
    # radius_down_bottom
    # height_up
    # height_down
    # grid_size
    # delta_x
    # delta_y
    # num_cones_up
    # num_cones_down
    # min_meshstep
    assert isinstance(num_cones_up, (int, type(None)))
    assert isinstance(num_cones_down, (int, type(None)))
    assert delta_x is None or delta_x == 0.0
    assert delta_y is None or delta_y == 0.0

    create_cones_up = True
    create_cones_down = True

    if delta_x is None:
        delta_x = 0.0
    if delta_y is None:
        delta_y = 0.0

    if radius_up_top is None:
        assert radius_up_bottom is None
        assert height_up is None
        assert num_cones_up is None
        radius_up_top = 0.0
        radius_up_bottom = 0.0
        height_up = 0.0
        num_cones_up = 0

        create_cones_up = False

    if radius_down_top is None:
        assert radius_down_bottom is None
        assert height_down is None
        assert num_cones_down is None
        radius_down_top = 0.0
        radius_down_bottom = 0.0
        height_down = 0.0
        num_cones_down = 0

        create_cones_down = False

    fun_round = np.round

    thickness_silver = fun_round(thickness_silver)
    thickness_up = fun_round(thickness_up)
    thickness_down = fun_round(thickness_down)
    radius_up_top = fun_round(radius_up_top)
    radius_up_bottom = fun_round(radius_up_bottom)
    radius_down_top = fun_round(radius_down_top)
    radius_down_bottom = fun_round(radius_down_bottom)
    height_up = fun_round(height_up)
    height_down = fun_round(height_down)
    delta_x = fun_round(delta_x)
    delta_y = fun_round(delta_y)

    if create_cones_up:
        grid_size_up = grid_size / num_cones_up
    else:
        grid_size_up = grid_size

    if create_cones_down:
        grid_size_down = grid_size / num_cones_down
    else:
        grid_size_down = grid_size

    utils_common.print_info(
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
    )

    assert constants.range_X_to_verify[0, 0] <= thickness_silver and thickness_silver <= constants.range_X_to_verify[0, 1]
    assert constants.range_X_to_verify[1, 0] <= thickness_up and thickness_up <= constants.range_X_to_verify[1, 1]
    assert constants.range_X_to_verify[2, 0] <= thickness_down and thickness_down <= constants.range_X_to_verify[2, 1]

    if create_cones_up:
        assert constants.range_X_to_verify[3, 0] <= radius_up_top and radius_up_top <= constants.range_X_to_verify[3, 1]
        assert constants.range_X_to_verify[4, 0] <= radius_up_bottom and radius_up_bottom <= constants.range_X_to_verify[4, 1]
        assert constants.range_X_to_verify[7, 0] <= height_up and height_up <= constants.range_X_to_verify[7, 1]
        assert constants.range_X_to_verify[9, 0] <= grid_size_up and grid_size_up <= constants.range_X_to_verify[9, 1]
        assert constants.range_X_to_verify[11, 0] <= num_cones_up and num_cones_up <= constants.range_X_to_verify[11, 1]

        assert radius_up_top < radius_up_bottom
        assert 2 * radius_up_bottom <= grid_size_up

    if create_cones_down:
        assert constants.range_X_to_verify[5, 0] <= radius_down_top and radius_down_top <= constants.range_X_to_verify[5, 1]
        assert constants.range_X_to_verify[6, 0] <= radius_down_bottom and radius_down_bottom <= constants.range_X_to_verify[6, 1]
        assert constants.range_X_to_verify[8, 0] <= height_down and height_down <= constants.range_X_to_verify[8, 1]
        assert constants.range_X_to_verify[10, 0] <= grid_size_down and grid_size_down <= constants.range_X_to_verify[10, 1]
        assert constants.range_X_to_verify[12, 0] <= num_cones_down and num_cones_down <= constants.range_X_to_verify[12, 1]

        assert radius_down_top > radius_down_bottom
        assert 2 * radius_down_top <= grid_size_down

    if create_cones_up and create_cones_down:
        assert np.abs(num_cones_up * grid_size_up - num_cones_down * grid_size_down) < 1e-8

    assert grid_size_down > delta_x
    assert grid_size_down > delta_y

    str_info = f'{min_meshstep:.1f}_{thickness_silver:.1f}_{thickness_up:.1f}_{thickness_down:.1f}_{radius_up_top:.1f}_{radius_up_bottom:.1f}_{radius_down_top:.1f}_{radius_down_bottom:.1f}_{height_up:.1f}_{height_down:.1f}_{grid_size_up:.1f}_{grid_size_down:.1f}'

    thickness_silver *= NANO
    thickness_up *= NANO
    thickness_down *= NANO
    radius_up_top *= NANO
    radius_up_bottom *= NANO
    radius_down_top *= NANO
    radius_down_bottom *= NANO
    height_up *= NANO
    height_down *= NANO
    grid_size *= NANO
    grid_size_up *= NANO
    grid_size_down *= NANO
    delta_x *= NANO
    delta_y *= NANO
    min_meshstep *= NANO

    half_thickness_silver = 0.5 * thickness_silver

    z_min_up = half_thickness_silver
    z_max_up = half_thickness_silver + thickness_up

    z_min_down = -half_thickness_silver - thickness_down
    z_max_down = -half_thickness_silver

    pitch = grid_size

    width_simulation = pitch
    width_plane = 2 * width_simulation

    dist_source_structure = 8 * min_meshstep
    dist_monitor_structure_transparency = 4 * min_meshstep
    dist_monitor_structure_reflection = 4 * min_meshstep
    dist_monitor_simulation_boundary = 2 * min_meshstep

    z_plane = z_max_up + height_up + dist_source_structure
    z_monitor_transparency = z_min_down - height_down - dist_monitor_structure_transparency
    z_monitor_reflection = z_plane + dist_monitor_structure_reflection

    sim_max = z_monitor_reflection + dist_monitor_simulation_boundary
    sim_min = z_monitor_transparency - dist_monitor_simulation_boundary

    time_start_creation = time.time()
    fdtd = lumapi.FDTD(hide=not debug)

    utils_common.print_separator()
    print('FDTD initialized.', flush=True)
    utils_common.print_separator()
    print('', flush=True)

    fdtd.setview("zoom", 40)
    fdtd.importmaterialdb("../materials/material_data.mdf")

    set_plate(fdtd, "up", z_min_up, z_max_up, width_simulation)
    set_plate(fdtd, "down", z_min_down, z_max_down, width_simulation)

    fdtd.addrect()
    fdtd.set("name", "silver")
    fdtd.set("x", 0.0)
    fdtd.set("y", 0.0)
    fdtd.set("z", 0.0)
    fdtd.set("x span", width_simulation)
    fdtd.set("y span", width_simulation)
    fdtd.set("z span", thickness_silver)
    fdtd.set("material", "Ag (Silver) - CRC")

    if create_cones_up:
        set_cones(fdtd, "up", pitch, grid_size_up, radius_up_bottom, radius_up_top, height_up, z_max_up, delta_x=delta_x, delta_y=delta_y)
    if create_cones_down:
        set_cones(fdtd, "down", pitch, grid_size_down, radius_down_bottom, radius_down_top, height_down, z_min_down)

    fdtd.addplane()
    fdtd.set("name", "source")

    fdtd.addpower()
    fdtd.set("name", "transparency_monitor")

    fdtd.addpower()
    fdtd.set("name", "reflection_monitor")

    fdtd.addfdtd()

    fdtd.addmesh()
    fdtd.set("name", "mesh_silver")

    fdtd.addmesh()
    fdtd.set("name", "mesh_up")

    fdtd.addmesh()
    fdtd.set("name", "mesh_down")

    if create_cones_up:
        fdtd.addmesh()
        fdtd.set("name", "mesh_cones_up")

    if create_cones_down:
        fdtd.addmesh()
        fdtd.set("name", "mesh_cones_down")

    configuration = [

        ("source",
            (
                ("injection axis", "z"),
                ("direction", "backward"),
                ("x", 0.0),
                ("y", 0.0),
                ("z", z_plane),
                ("X span", width_plane),
                ("y span", width_plane),
                ("wavelength start", 400e-9), # in meters
                ("wavelength stop", 700e-9), # in meters
            )
        ),

        ("transparency_monitor",
            (
                ("monitor type", 7),
                ("x", 0.0),
                ("y", 0.0),
                ("z", z_monitor_transparency),
                ("X span", width_simulation),
                ("y span", width_simulation),
                ("override global monitor settings", True),
                ("use wavelength spacing", True),
                ("frequency points", 300),
            )
        ),

        ("reflection_monitor",
            (
                ("monitor type", 7),
                ("x", 0.0),
                ("y", 0.0),
                ("z", z_monitor_reflection),
                ("X span", width_simulation),
                ("y span", width_simulation),
                ("override global monitor settings", True),
                ("use wavelength spacing", True),
                ("frequency points", 300),
            )
        ),

        ("FDTD",
            (
                ("dimension", "3D"),
                ("simulation time", 150e-15), # in seconds
                ("simulation temperature", 300), # K
                ("x", 0.0),
                ("y", 0.0),
                ("x span", width_simulation),
                ("y span", width_simulation),
                ("z min", sim_min),
                ("z max", sim_max),
                ("mesh refinement", 3),
                ("x min bc", "Periodic"),
                ("x max bc", "Periodic"),
                ("y min bc", "Periodic"),
                ("y max bc", "Periodic"),
                ("z min bc", "PML"),
                ("z max bc", "PML"),
                ("Mesh accuracy", MESH_ACCURACY),
                ("pml profile", 1),
                ("pml profile", 4),
                ("pml layers", 48),
                ("pml alpha", 0.1),
                ("auto shutoff min", AUTOSHUTOFF),
                ("Min mesh step", min_meshstep),

            )
        ),

        ("mesh_silver",
            (
                ("based on a structure", True),
                ("structure", "silver"),
                ("set maximum mesh step", True),
                ("override x mesh", True),
                ("override y mesh", True),
                ("override z mesh", True),
                ("dx", min_meshstep),
                ("dy", min_meshstep),
                ("dz", dz_silver),
            )
        ),

        ("mesh_up",
            (
                ("based on a structure", True),
                ("structure", "highindex_up"),
                ("set maximum mesh step", True),
                ("override x mesh", True),
                ("override y mesh", True),
                ("override z mesh", True),
                ("dx", min_meshstep),
                ("dy", min_meshstep),
                ("dz", dz_up),
            )
        ),

        ("mesh_down",
            (
                ("based on a structure", True),
                ("structure", "highindex_down"),
                ("set maximum mesh step", True),
                ("override x mesh", True),
                ("override y mesh", True),
                ("override z mesh", True),
                ("dx", min_meshstep),
                ("dy", min_meshstep),
                ("dz", dz_down),
            )
        ),

    ]

    if create_cones_up:
        configuration.append(
            ("mesh_cones_up",
                (
                    ("directly defined", True),
                    ("structure", "cones_up"),
                    ("x", 0.0),
                    ("y", 0.0),
                    ("z", z_max_up + 0.5 * height_up),
                    ("x span", width_simulation),
                    ("y span", width_simulation),
                    ("z span", height_up),
                    ("set maximum mesh step", True),
                    ("override x mesh", True),
                    ("override y mesh", True),
                    ("override z mesh", True),
                    ("dx", min_meshstep),
                    ("dy", min_meshstep),
                    ("dz", min_meshstep),
                )
            )
        )

    if create_cones_down:
        configuration.append(
            ("mesh_cones_down",
                (
                    ("directly defined", True),
                    ("structure", "cones_down"),
                    ("x", 0.0),
                    ("y", 0.0),
                    ("z", z_min_down - 0.5 * height_down),
                    ("x span", width_simulation),
                    ("y span", width_simulation),
                    ("z span", height_down),
                    ("set maximum mesh step", True),
                    ("override x mesh", True),
                    ("override y mesh", True),
                    ("override z mesh", True),
                    ("dx", min_meshstep),
                    ("dy", min_meshstep),
                    ("dz", min_meshstep),
                )
            )
        )

    for obj, parameters in configuration:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)

    time_end_creation = time.time()
    time_to_create = time_end_creation - time_start_creation

    str_models = '../models'
    if not os.path.exists(str_models):
        os.mkdir(str_models)

    str_now = utils_common.get_str_now()
    str_file = f'exp_{str_info}_{str_now}'
    str_file = str_file.replace('.', '_')
    str_file = os.path.join(str_models, str_file)

    fdtd.save(str_file)

    utils_common.print_separator()
    print(f'Time to create: {time_to_create:.4f} sec.', flush=True)
    time_start_run = time.time()
    fdtd.run()
    time_end_run = time.time()

    time_to_simulate = time_end_run - time_start_run
    print(f'Time to simulate: {time_to_simulate:.4f} sec.', flush=True)

    results_transparency = fdtd.getresult('transparency_monitor', 'T')
    results_reflection = fdtd.getresult('reflection_monitor', 'T')

    emi_se = shielding_effectiveness.compute_se(thickness_silver)

    results_transparency_T = results_transparency['T']
    print(results_transparency_T.shape, flush=True)
    mean_results_transparency = np.mean(results_transparency_T) * -1.0

    results_reflection_T = results_reflection['T']
    print(results_reflection_T.shape, flush=True)
    mean_results_reflection = np.mean(results_reflection_T)

    print(f'mean_results_transparency {mean_results_transparency:.4f}', flush=True)
    print(f'mean_results_reflection {mean_results_reflection:.4f}', flush=True)
    print(f'emi_se {emi_se:.4f}', flush=True)
    print('', flush=True)

    if debug:
        time.sleep(1e6)

    fdtd.close()
    if os.path.exists(str_file + '.fsp'):
        os.remove(str_file + '.fsp')
    if os.path.exists(str_file + '_p0.log'):
        os.remove(str_file + '_p0.log')

    return mean_results_transparency, results_transparency_T, mean_results_reflection, results_reflection_T, emi_se, time_to_create, time_to_simulate


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    debug = args.debug

    thickness_up = 20
    thickness_down = 25
    thickness_silver = 10

    radius_up_top = 20
    radius_up_bottom = 25

    radius_down_top = 21
    radius_down_bottom = 10

    height_up = 55
    height_down = 52

    grid_size = 57

    delta_x = 0
    delta_y = 0

    num_cones_up = 1
    num_cones_down = 1

    min_meshstep = 1

    mean_transparency, transparency, mean_reflection, reflection, emi_se, time_to_create, time_to_simulate = run_simulation(
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
        delta_x,
        delta_y,
        num_cones_up,
        num_cones_down,
        min_meshstep,
        debug=debug,
    )

    print(mean_transparency)
    print(mean_reflection)
    print(emi_se)
    print(time_to_create)
    print(time_to_simulate)

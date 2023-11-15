import numpy as np
import os
import argparse
import time

import bo
import objective
import constants
import utils_common
import utils_hypervolumes


path_results = constants.path_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--str_fun', type=str, required=True)
    parser.add_argument('--num_iter_low', type=int, required=True)
    parser.add_argument('--num_iter_high', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)

    args = parser.parse_args()
    str_fun = args.str_fun
    str_method = 'ours'
    num_iter_low = args.num_iter_low
    num_iter_high = args.num_iter_high
    seed = args.seed

    use_time = False
    time_until = None
    use_time_low = False
    assert not (num_iter_low == -1 and num_iter_high > 0)

    if num_iter_low == -1:
        use_time_low = True
        num_iter_low = 10000000
        time_low = 0.2

    if num_iter_high == -1:
        time_budgets_target = constants.time_budgets_target

        for str_tar, tim in time_budgets_target:
            if str_tar == str_fun:
                time_until = tim

        print(f'time_until {time_until}')
        print('')
        assert time_until is not None
        num_iter_high = 10000000
        use_time = True
        time_margin = 1.05

    num_init_low = 8

    high_fidelity = 2
    low_fidelity = 40

    obj_high = objective.Objective(str_fun, high_fidelity)
    obj_low = objective.Objective(str_fun, low_fidelity)

    assert np.all(obj_high.bounds == obj_low.bounds)
    bounds = obj_high.bounds
    time_start = time.time()

    if not use_time and not use_time_low:
        str_file = f'multiple_wavelengths_{str_fun}_{str_method}_init_{num_init_low}_iter_{num_iter_low}_{num_iter_high}_fidelity_{low_fidelity}_{high_fidelity}_seed_{seed}.npy'
    elif use_time and not use_time_low:
        str_file = f'multiple_wavelengths_{str_fun}_{str_method}_init_{num_init_low}_iter_{num_iter_low}_time_{time_until}_fidelity_{low_fidelity}_{high_fidelity}_seed_{seed}.npy'
    elif use_time and use_time_low:
        str_file = f'multiple_wavelengths_{str_fun}_{str_method}_init_{num_init_low}_iter_time_{time_low * time_until}_time_{time_until}_fidelity_{low_fidelity}_{high_fidelity}_seed_{seed}.npy'
    else:
        raise ValueError
    print(f'str_file {str_file}')

    ##
    model_bo_low = bo.BO(bounds, str_cov='matern52', str_acq='ei')

    X_low = model_bo_low.get_initials('sobol', num_init_low)
    for bx in X_low:
        obj_low.transparency(bx)
    Y_low = np.array([obj_low.transparencies, obj_low.efficiencies]).T
    print(f'X_low.shape {X_low.shape} Y_low.shape {Y_low.shape}')

    for ind_low in range(0, num_iter_low):
        next_point, _ = model_bo_low.optimize(X_low, Y_low, seed=seed * (ind_low + 1))
        obj_low.transparency(next_point)

        X_low = np.array(obj_low.queries)
        Y_low = np.array([obj_low.transparencies, obj_low.efficiencies]).T
        print(f'X_low.shape {X_low.shape} Y_low.shape {Y_low.shape}')

        time_now = time.time()
        if use_time and use_time_low and (time_now - time_start) > time_low * time_until:
            print(time_now - time_start)
            break

    is_pareto = utils_hypervolumes.is_pareto_frontiers(-1.0 * Y_low)
    pareto_frontiers = X_low[is_pareto]
    print(f'num_pareto_frontiers {pareto_frontiers.shape[0]}')

    ##
    model_bo_high = bo.BO(bounds, str_cov='matern52', str_acq='ei')

    indices_high = np.random.RandomState(seed + 1001).choice(pareto_frontiers.shape[0],
        np.minimum(pareto_frontiers.shape[0], 10),
        replace=False)
    num_init_high = indices_high.shape[0]
    print(f'num_init_high {num_init_high}')

    X_high = pareto_frontiers[indices_high]
    for bx in X_high:
        obj_high.transparency(bx)
    Y_high = np.array([obj_high.transparencies, obj_high.efficiencies]).T
    print(f'X_high.shape {X_high.shape} Y_high.shape {Y_high.shape}')

    for ind_high in range(0, num_iter_high):
        next_point, _ = model_bo_high.optimize(X_high, Y_high, seed=seed * (ind_high + 1))
        obj_high.transparency(next_point)

        X_high = np.array(obj_high.queries)
        Y_high = np.array([obj_high.transparencies, obj_high.efficiencies]).T
        print(f'X_high.shape {X_high.shape} Y_high.shape {Y_high.shape}')

        time_now = time.time()
        if use_time and (time_now - time_start) > time_margin * time_until:
            print(time_now - time_start)
            break

    queries_high = obj_high.queries
    transparencies_high = obj_high.transparencies
    efficiencies_high = obj_high.efficiencies
    times_high = obj_high.times
    datetimes_start_high = obj_high.datetimes_start
    datetimes_end_high = obj_high.datetimes_end

    queries_low = obj_low.queries
    transparencies_low = obj_low.transparencies
    efficiencies_low = obj_low.efficiencies
    times_low = obj_low.times
    datetimes_start_low = obj_low.datetimes_start
    datetimes_end_low = obj_low.datetimes_end

    dict_all = {
        'str_fun': str_fun,
        'str_method': str_method,
        'num_init_low': num_init_low,
        'num_init_high': num_init_high,
        'num_iter_low': num_iter_low,
        'num_iter_high': num_iter_high,
        'ind_high': ind_high,
        'use_time': use_time,
        'time_until': time_until,
        'seed': seed,
        'queries_high': queries_high,
        'transparencies_high': transparencies_high,
        'efficiencies_high': efficiencies_high,
        'times_high': times_high,
        'datetimes_start_high': datetimes_start_high,
        'datetimes_end_high': datetimes_end_high,
        'queries_low': queries_low,
        'transparencies_low': transparencies_low,
        'efficiencies_low': efficiencies_low,
        'times_low': times_low,
        'datetimes_start_low': datetimes_start_low,
        'datetimes_end_low': datetimes_end_low,
    }

    len_high_fidelity = len(queries_high)
    len_low_fidelity = len(queries_low)

    assert (len_high_fidelity + len_low_fidelity) == (num_init_high + num_init_low + ind_high + 1 + ind_low + 1)

    print('')
    utils_common.print_separator()
    print(f'len_high_fidelity {len_high_fidelity}')
    print(f'len_low_fidelity {len_low_fidelity}')
    utils_common.print_separator()

    if not os.path.exists(path_results):
        os.mkdir(path_results)
    np.save(os.path.join(path_results, str_file), dict_all)

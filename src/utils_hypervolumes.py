import numpy as np

import shielding_effectiveness


def is_pareto_frontiers(objs):
    assert isinstance(objs, np.ndarray)
    assert len(objs.shape) == 2
    assert objs.shape[1] == 2

    is_pareto = np.ones(objs.shape[0], dtype=bool)

    for i, c in enumerate(objs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(objs[is_pareto] > c, axis=1)
            is_pareto[i] = True
    return is_pareto

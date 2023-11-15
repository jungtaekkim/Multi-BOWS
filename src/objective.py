import numpy as np
import os
import time

import wrappers_nanocones
import utils_common
import constants


class Objective:
    def __init__(self, str_fun, min_meshstep, obj_additional=None):
        self.str_fun = str_fun
        self.min_meshstep = min_meshstep

        if str_fun == 'threelayer':
            self.bounds = constants.range_X[:3, :]
            self.function = lambda bx: wrappers_nanocones.fun_target_threelayer(bx, min_meshstep)
        elif str_fun == 'matched':
            self.bounds = constants.range_X[:10, :]
            self.function = lambda bx: wrappers_nanocones.fun_target_matched(bx, min_meshstep)
        elif str_fun == 'unmatched':
            self.bounds = constants.range_X
            self.function = lambda bx: wrappers_nanocones.fun_target_unmatched(bx, min_meshstep)
        elif str_fun == 'automatic':
            self.bounds = np.concatenate([constants.range_X, np.array([[-0.5001, 4.4999]])], axis=0)
            self.function = lambda bx: wrappers_nanocones.fun_target_automatic(bx, min_meshstep)
        else:
            raise ValueError

        self.queries = []
        self.transparencies = []
        self.efficiencies = []
        self.times = []
        self.datetimes_start = []
        self.datetimes_end = []

        self.obj_additional = obj_additional

    def run(self, bx):
        str_now_start = utils_common.get_str_now()
        time_start = time.time()

        mean_transparencies, transparencies, mean_reflections, reflections, emi_se, time_to_create, time_to_simulate = self.function(bx)
        mean_transparencies *= -1.0
        emi_se *= -1.0
        time_end = time.time()
        str_now_end = utils_common.get_str_now()

        self.queries.append(bx)
        self.transparencies.append(mean_transparencies)
        self.efficiencies.append(emi_se)
        self.times.append(time_end - time_start)
        self.datetimes_start.append(str_now_start)
        self.datetimes_end.append(str_now_end)
        assert len(self.queries) == len(self.transparencies) == len(self.efficiencies) == len(self.times)

        if self.obj_additional is not None:
            self.obj_additional.run(bx)

        return mean_transparencies, emi_se

    def transparency(self, bx):
        trans, _ = self.run(bx)

        return trans

    def shielding_efficiency(self, bx):
        _, effi = self.run(bx)

        return effi

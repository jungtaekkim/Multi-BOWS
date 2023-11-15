import numpy as np


CONDUCTIVITY_SILVER = 6.3e7 # Siemens/m
# https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity

NANO = 1e-9


def compute_se(thickness_silver):
    Rs = 1.0 / (thickness_silver * CONDUCTIVITY_SILVER)
    emi_se = 20.0 * np.log10(1.0 + 377.0 / (Rs * 2))

    return emi_se


if __name__ == '__main__':
    list_thickness_silver = np.arange(3, 21) * NANO

    for thickness_silver in list_thickness_silver:
        print(compute_se(thickness_silver))

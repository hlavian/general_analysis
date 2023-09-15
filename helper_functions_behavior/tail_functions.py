import numpy as np


def nan_phase_jumps(phase_array):
    out_array = phase_array.copy()
    out_array[1:][np.abs(np.diff(out_array)) > np.pi] = np.nan
    return out_array

import numpy as np


def normalize_traces(traces):

    norm_traces = np.copy(traces)
    norm_traces = norm_traces.T # need to transpose it since the functions work like that
    sd = np.nanstd(norm_traces)
    mean = np.nanmean(norm_traces)
    norm_traces = norm_traces-mean #numerator in the formula for z-score
    norm_traces = norm_traces/sd
    norm_traces = norm_traces.T
    return norm_traces


import numpy as np
from .constants import *

def scale_btd(data):
    data_scaled_01 = (data - norm_constants.btd_vmin)/(norm_constants.btd_vmax-norm_constants.btd_vmin)
    data_scaled01_inv = 1.001-data_scaled_01
    data_scaled01_inv = np.maximum(data_scaled01_inv, 1.e-6)
    data_scaled01_inv_log = np.log(data_scaled01_inv)
    data_scaled2 = 1. - (data_scaled01_inv_log - np.log(0.001))/(-np.log(0.001))
    return data_scaled2

def scale_btd_back(data):
    data_scaled01_inv_log = (1. - data)*(-np.log(0.001)) + np.log(0.001)
    data_exp = np.exp(data_scaled01_inv_log)
    data_exp_inv = 1.001-data_exp
    data_descaled = data_exp_inv*(norm_constants.btd_vmax-norm_constants.btd_vmin) + norm_constants.btd_vmin
    return data_descaled


def scale_ch5(data):
    # ch5_vmin = 205.
    # ch5_vmax = 260.
    return 1 + (norm_constants.ch5_vmin - data)/(norm_constants.ch5_vmax - norm_constants.ch5_vmin)

def scale_ch5_back(data):
    # ch5_vmin = 205.
    # ch5_vmax = 260.
    return norm_constants.ch5_vmin - (data-1.)*(norm_constants.ch5_vmax-norm_constants.ch5_vmin)


def scale_ch9(data):
    return np.minimum(1. + (norm_constants.ch9_vmin - data)/(norm_constants.ch9_vmax - norm_constants.ch9_vmin), 1.)

def scale_ch9_back(data):
    return norm_constants.ch9_vmin - (data-1.)*(norm_constants.ch9_vmax-norm_constants.ch9_vmin)

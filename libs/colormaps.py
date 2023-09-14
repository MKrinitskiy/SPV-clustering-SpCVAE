from .constants import *

def create_ch9_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # ch9_vmin = 200.
    # ch9_vmax = 320.
    # ch9_vm1 = 227.
    jet_cnt = int(512 * (norm_constants.ch9_thresh - norm_constants.ch9_vmin) / (norm_constants.ch9_vmax - norm_constants.ch9_vmin))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([jetcolors[::-1], graycolors], axis=0)
    ch9_cm = ListedColormap(newcolors)
    return ch9_cm



def create_btd_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # btd_vmin = -80
    # btd_vmax = 5.5
    # btd_vm1 = 0.
    jet_cnt = int(512 * (norm_constants.btd_vmax - norm_constants.btd_thresh) / (norm_constants.btd_vmax - norm_constants.btd_vmin))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([graycolors[::-1], jetcolors], axis=0)
    btd_cm = ListedColormap(newcolors)
    return btd_cm


def create_ch5_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # ch5_vmin = 205.
    # ch5_vmax = 260.
    # ch5_vm1 = 223.
    # ch5_vm1 = 237.

    jet_cnt = int(512 * (norm_constants.ch5_thresh - norm_constants.ch5_vmin) / (norm_constants.ch5_vmax - norm_constants.ch5_vmin))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([jetcolors[::-1], graycolors[::-1]], axis=0)
    newcm = ListedColormap(newcolors)
    return newcm


def create_ch5_scaled_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    from .scaling import scale_ch5

    thresh = scale_ch5(norm_constants.ch5_thresh)

    jet_cnt = int(512 * (1.-thresh))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([graycolors[::-1], jetcolors], axis=0)
    newcm = ListedColormap(newcolors)
    return newcm


def create_ch9_scaled_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    from libs.scaling import scale_ch9

    ch9_vm1 = norm_constants.ch9_thresh
    thresh = scale_ch9(ch9_vm1)

    jet_cnt = int(512 * (1. - thresh))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([graycolors[::-1], jetcolors], axis=0)
    ch9_cm = ListedColormap(newcolors)
    return ch9_cm


def create_btd_scaled_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    from .scaling import scale_btd

    btd_vm1 = norm_constants.btd_thresh
    thresh = scale_btd(btd_vm1)

    jet_cnt = int(512 * (1.-thresh))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([graycolors[::-1], jetcolors], axis=0)
    btd_cm = ListedColormap(newcolors)
    return btd_cm
import os
import numpy as np
import fnmatch
# import cv2
# from scipy.misc import imresize
import sys, traceback
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import math_ops
import pdb
from pathlib import Path


def replicate_array_to_depth3(arr):
    x = np.expand_dims(arr, axis=2)
    x = np.repeat(x, 3, axis=2)
    return x




def DoesPathExistAndIsDirectory(pathStr):
    if os.path.exists(pathStr) and os.path.isdir(pathStr):
        return True
    else:
        return False



def EnsureDirectoryExists(pathStr):
    if not DoesPathExistAndIsDirectory(pathStr):
        try:
            # os.mkdir(pathStr)
            Path(pathStr).mkdir(parents=True, exist_ok=True)
        except Exception as ex:
            err_fname = './errors.log'
            exc_type, exc_value, exc_traceback = sys.exc_info()
            with open(err_fname, 'a') as errf:
                traceback.print_tb(exc_traceback, limit=None, file=errf)
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=errf)
            print(str(ex))
            print('the directory you are trying to place a file to doesn\'t exist and cannot be created:\n%s' % pathStr)
            raise FileNotFoundError('the directory you are trying to place a file to doesn\'t exist and cannot be created:')



def find_files(directory, pattern, maxdepth=None):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\\\', os.sep)
                if maxdepth is None:
                    yield filename
                else:
                    if filename.count(os.sep)-directory.count(os.sep) <= maxdepth:
                        yield filename


def find_directories(directory, pattern=None, maxdepth=None):
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if pattern is None:
                retname = os.path.join(root, d, '')
                yield retname
            elif fnmatch.fnmatch(d, pattern):
                retname = os.path.join(root, d, '')
                retname = retname.replace('\\\\', os.sep)
                if maxdepth is None:
                    yield retname
                else:
                    if retname.count(os.sep)-directory.count(os.sep) <= maxdepth:
                        yield retname


def tensor_int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape

    try:
        return K.int_shape(x)
    except:
        pass

    try:
        return tuple(x.get_shape().as_list())
    except ValueError:
        return None




def custom_MSE(y_true, y_pred):
    if len(tensor_int_shape(y_true)) == 2:
        return K.mean(K.zeros_like(y_true), axis=-1)
    # return K.mean(K.square(y_pred - y_true), axis=-1)
    return K.mean(K.square(y_pred - y_true), axis=[-1,-2])


def sampling(sample_args):
    """ sample latent layer from normal prior
    """
    z_mean, z_log_var = sample_args
    dim = K.int_shape(z_mean)[1]
    batch = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



def expand1depth_to_3depth_tensor(x):
    x2 = K.concatenate([x,x,x], -1)
    return x2


def vae_KL_loss(y_true, y_pred):
    vec_len = int(tensor_int_shape(y_pred)[1]/2)
    z_mean = y_pred[:,:vec_len]
    z_log_var = y_pred[:,vec_len:]

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss



class SpVAE_loss(object):
    def __init__(self, rho=0.05, beta=1.0, debug = False, logfile = './SpVAE_loss.log'):
        self.__name__ = "SpVAE_loss"
        self.rho = rho
        self.beta = beta
        self.logfile = logfile
        self.debug = debug


    def __call__(self, *args, **kwargs):
        # y_true = args[0]
        y_pred = args[1]
        vec_len = int(tensor_int_shape(y_pred)[1] / 2)
        z_mean = y_pred[:, :vec_len]
        z_log_var = y_pred[:, vec_len:]

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5

        # rho = 0.05
        # beta = 1.0
        data_rho = K.mean(z_mean, 0)
        reg_cost = -K.mean(self.rho * K.log(data_rho / self.rho + 1.e-9) + (1 - self.rho) * K.log(1.e-9 + (1 - data_rho) / (1 - self.rho)))

        # if self.debug:
        #     pdb.set_trace()

        # if tf.is_nan(reg_cost):
        #     if self.debug:
        #         with open(self.logfile, 'a') as f:
        #             f.writelines('reg_cost is NaN!!! rho = %e; data_rho = %e' % (self.rho, data_rho))
        #             f.writelines('z_mean = %s' % str(z_mean))
        #     reg_cost = 0.0
        def f1():
            tf.print(reg_cost, [reg_cost], 'reg_cost is nan')
            return tf.zeros_like(reg_cost)
        def f2(): return reg_cost
        reg_cost = tf.cond(tf.math.is_nan(reg_cost), f1, f2)

        cost = kl_loss + self.beta * reg_cost

        return cost

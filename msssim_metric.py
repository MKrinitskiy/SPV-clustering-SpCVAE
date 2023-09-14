from __future__ import absolute_import
# from keras.objectives import *
import tensorflow.keras.backend as K
# import keras_contrib.backend as KC
from tensorflow_backend import *
import tensorflow as tf
from support_defs import tensor_int_shape
import pdb, os


class MSSSIMMetric():
    def __init__(self, k1=0.01, k2=0.03, filter_size=3, max_val=1.0, weights=None, filter_sigma=1.5, average=True, debug = False, log_directory = './logs/', tag=''):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
        self.__name__ = 'MSSSIMMetric' + (('_%s' % tag) if len(tag)>0 else '')
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.max_val = max_val
        self.c1 = (self.k1 * self.max_val) ** 2
        self.c2 = (self.k2 * self.max_val) ** 2
        self.dim_ordering = K.image_data_format()
        self.weights = np.array(weights if weights else
                     [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        self.levels = self.weights.size
        self.downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
        self.gaussian = make_kernel(1.5)
        self.iterations = 5
        self.ms_ssim = []
        self.filter_sigma = filter_sigma
        self.average = average
        self.debug = debug
        if (len(tag)>0):
            self.logfile = os.path.join(log_directory, ('%s_%s' % (tag, 'MSSSIMMetric_debug.log')))
        else:
            self.logfile = os.path.join(log_directory, 'MSSSIMMetric_debug.log')
        self.tag = tag

    def __int_shape(self, x):
        if hasattr(x, '_keras_shape'):
            return x._keras_shape
        try:
            return tuple(x.get_shape().as_list())
        except ValueError:
            return None




    def ssim(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [self.filter_size, self.filter_size]

        # replace None by -1 in a 'shape'
        y_true = tf.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        # y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))
        # y_pred = tf.expand_dims(y_pred, -1)

        patches_pred = extract_image_patches(y_pred, kernel, kernel, 'valid', self.dim_ordering)
        patches_true = extract_image_patches(y_true, kernel, kernel, 'valid', self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = tf.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = tf.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = tf.reduce_mean(patches_true, axis=-1)
        u_pred = tf.reduce_mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        # return K.mean((1.0 - ssim) / 2.0)
        cs = (2 * covar_true_pred + self.c2)/(var_pred + var_true + self.c2)
        return K.mean(ssim, axis=[-1,-2]),K.mean(cs, axis=[-1,-2])





    def __call__(self, y_true, y_pred):

        if len(tensor_int_shape(y_true)) == 2:
            return K.mean(K.zeros_like(y_true), axis=-1)

        # mssim = np.array([])
        mssim = []
        # mcs = np.array([])
        mcs = []
        img1 = y_true
        # img1 = tf.expand_dims(img1, -1)
        img2 = y_pred
        # img2 = tf.expand_dims(img2, -1)
        for i in range(self.levels):
            ssim,cs = self.ssim(img1, img2)
            # mssim = np.append(mssim, ssim)
            mssim.append(ssim ** self.weights[i])
            # mcs = np.append(mcs, cs)
            mcs.append(cs ** self.weights[i])
            # img1 = tf.nn.conv2d(img1, self.gaussian, strides=[1, 1, 1, 1], padding='SAME')
            # img2 = tf.nn.conv2d(img2, self.gaussian, strides=[1, 1, 1, 1], padding='SAME')
            # filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            # filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            img1 = filtered_im1
            img2 = filtered_im2
            # img1 = K.resize_images(img1, 2, 2, 'channels_last')
            # img2 = K.resize_images(img2, 2, 2, 'channels_last')

        # mssim = tf.stack(list(mssim), axis=0)
        # mcs = tf.stack(list(mcs), axis=0)
        mssim = tf.stack(mssim, axis=-1)
        mcs = tf.stack(mcs, axis=-1)
        if self.average:
            # value = tf.reduce_mean((tf.reduce_prod(mcs ** self.weights, axis=-1) * (mssim ** self.weights)), axis=-1)
            value = tf.reduce_mean((tf.reduce_prod(mcs, axis=-1) * (mssim[:,self.levels-1])), axis=-1)
        else:
            # value = tf.reduce_mean((tf.reduce_prod(mcs ** self.weights, axis=-1) * (mssim ** self.weights)))
            value = (tf.reduce_prod(mcs, axis=-1) * (mssim[:,self.levels-1]))


        # if self.debug:
        #     pdb.set_trace()

        # if tf.is_nan(value):
        #     if self.debug:
        #         with open(self.logfile, 'a') as f:
        #             f.writelines('MSSSIM metric is NaN!!!')
        #             f.writelines('mssim = %s' % str(mssim))
        #             f.writelines('mcs = %s' % str(mcs))
        #     value = 0.0
        def f1():
            tf.print(value, [value], 'reg_cost is nan')
            return tf.zeros_like(value)
        def f2(): return value

        value = tf.cond(tf.math.is_nan(tf.reduce_sum(value)), f1, f2)


        return 1.0-value



# def KCmean(x, axis=None, keepdims=False):
#     return tf.reduce_mean(x, axis, keepdims)



def make_kernel(sigma):
    # kernel radius = 2*sigma, but minimum 3x3 matrix
    kernel_size = max(3, int(2 * 2 * sigma + 1))
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
    # make 2D kernel
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
    # normalize kernel by sum of elements
    kernel = np_kernel / np.sum(np_kernel)
    kernel = np.reshape(kernel, (kernel_size, kernel_size, 1,1))    #height, width, in_channels, out_channel
    return kernel


def gaussian(x, mu, sigma):
    return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))
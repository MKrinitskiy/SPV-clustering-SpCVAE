from __future__ import absolute_import
from tensorflow.keras import backend as K
from tensorflow_backend import *
import tensorflow as tf
from support_defs import tensor_int_shape



class SSIMMetric():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0, tag = 'output'):
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
        self.__name__ = 'SSIMMetric' + (('_%s' % tag) if len(tag)>0 else '')
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.tag = tag
        # self.backend = KC.backend()

    def __int_shape(self, x):
        if hasattr(x, '_keras_shape'):
            return x._keras_shape
        try:
            return tuple(x.get_shape().as_list())
        except ValueError:
            return None

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        if len(tensor_int_shape(y_true)) == 2:
            return K.mean(K.zeros_like(y_true), axis=-1)

        kernel = [self.kernel_size, self.kernel_size]

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
        u_true = tf.reduce_mean(patches_true, axis=-1, keepdims=False)
        u_pred = tf.reduce_mean(patches_pred, axis=-1, keepdims=False)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        # return K.mean((1.0 - ssim) / 2.0)
        return 1.0-K.mean(ssim, axis=[-1,-2])

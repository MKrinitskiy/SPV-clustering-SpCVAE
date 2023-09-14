import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img, random_channel_shift
import numpy as np
# import cv2
import fnmatch
from tensorflow.python.keras.callbacks import Callback
import tensorflow.keras as keras



class KFoldCycleCallback(Callback):

    def __init__(self, myImageDataGenerator, epochs_to_cycle=1):
        self.myImageDataGenerator = myImageDataGenerator
        self.epochs_to_cycle = epochs_to_cycle
        super(Callback, self).__init__()


    def on_epoch_begin(self, epoch, logs={}):
        if (epoch > 0) & (epoch % self.epochs_to_cycle == 0):
            self.myImageDataGenerator.cycle_KFold_train_val_sets()




class K_folds_iterator(Iterator):

    def __init__(self, x, batch_size=32):
        self.x = x
        super(K_folds_iterator, self).__init__(x.shape[0], batch_size, shuffle=False, seed=None)


    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        batch_x = self.x[index_array]
        return batch_x

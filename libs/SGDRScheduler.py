from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping, Callback
import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow as tf
import pickle


class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2,
                 # on_batch_end_fire_per_batches = 10,
                 logfile = './logs/lr_history.pkl'):


        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        # self.prev_batch_fired = 0
        # self.on_batch_end_fire_per_batches = on_batch_end_fire_per_batches

        self.current_batch = 0
        self.current_epoch = 0
        self.lr_started_being_logged = False
        self.logfile = logfile
        self.lr_to_set = self.max_lr

        self.history = {}

        super().__init__()



    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)
        # self.log_lr_value()

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''

        # if batch-self.prev_batch_fired < self.on_batch_end_fire_per_batches:
        #     return
        # self.prev_batch_fired = batch

        self.current_batch = batch

        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        self.lr_to_set = self.clr()
        K.set_value(self.model.optimizer.lr, self.lr_to_set)
        # self.log_lr_value()
        logs['lr'] = self.lr_to_set

    def on_epoch_begin(self, epoch, logs=None):
        self.prev_batch_fired = 0
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            # self.best_weights = self.model.get_weights()

        with open(self.logfile, 'wb') as f:
            pickle.dump(self.history, f)


    # def on_train_end(self, logs={}):
    #     '''Set weights to the values from the end of the most recent cycle for best performance.'''
    #     self.model.set_weights(self.best_weights)


    # def log_lr_value(self):
    #     if not self.lr_started_being_logged:
    #         with open(self.logfile, 'a') as f:
    #             f.write('epoch;batch;lr\n')
    #         self.lr_started_being_logged = True
    #
    #     with open(self.logfile, 'a') as f:
    #         f.write('%d;%d;%e\n' % (self.current_epoch, self.current_batch, self.lr_to_set))

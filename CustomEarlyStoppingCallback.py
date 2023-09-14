from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.platform import tf_logging as logging


class CustomEarlyStopping(EarlyStopping):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None):
        if callable(monitor):
            super(CustomEarlyStopping, self).__init__('accuracy', min_delta, patience, verbose, mode, baseline)
            self.monitor = monitor
        else:
            super(CustomEarlyStopping, self).__init__(monitor, min_delta, patience, verbose, mode, baseline)



    def on_epoch_end(self, epoch, logs=None):
        if callable(self.monitor):
            current = self.monitor(logs)
        else:
            current = logs.get[self.monitor]

        if current is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True



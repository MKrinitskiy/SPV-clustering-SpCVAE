from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging


class LogLearningRateCallback(Callback):
    def __init__(self, model, log_filename):
        super(LogLearningRateCallback, self).__init__()
        self.model = model
        self.log_filename = log_filename



    def on_epoch_end(self, epoch, logs=None):
        curr_lr_value = self.model.optimizer.learning_rate(epoch)
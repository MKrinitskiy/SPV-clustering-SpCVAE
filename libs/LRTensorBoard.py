from tensorflow.python.keras.callbacks import Callback, TensorBoard
from tensorflow.python.keras import backend as K

class LRTensorBoard(TensorBoard):
    def __init__(self, log_lr_every_n_batch=10, **kwargs):  # add other arguments to __init__ if you need
        super(LRTensorBoard, self).__init__(**kwargs)
        self.log_lr_every_n_batch = log_lr_every_n_batch
        self.batches_passed = 0

    def on_batch_end(self, batch, logs=None):
        super(LRTensorBoard, self).on_batch_end(batch, logs)

        self.batches_passed += 1
        if self.batches_passed % self.log_lr_every_n_batch == 0:
            logs['lr'] = K.get_value(self.model.optimizer.lr)

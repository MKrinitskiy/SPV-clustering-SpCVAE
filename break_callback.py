from tensorflow.python.keras.callbacks import Callback
import os




class BreakLearningCallback(Callback):
    def __init__(self, model, control_file = './break/triggerfile'):
        self.model = model
        self.control_file = control_file
        super(BreakLearningCallback, self).__init__()


    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists(self.control_file):
            self.model.stop_training = True
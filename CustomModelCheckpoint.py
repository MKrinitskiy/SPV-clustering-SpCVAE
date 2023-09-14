from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import os
from libs.service_defs import isSequence
from collections import Sequence, Hashable
from shutil import copyfile


class CustomModelCheckpoint(Callback):
    def __init__(self, model_to_save, filepath_monitors, filepath_overall, monitors=('val_accuracy'), verbose=0, modes=('auto'), start_epoch=0):
        assert np.all([isinstance(m, Hashable) for m in monitors]) # we use each monitor as a dict key so it should be hashable
        super(CustomModelCheckpoint, self).__init__()

        self.model_to_save = model_to_save
        self.verbose = verbose
        self.monitors = monitors
        self.modes = {}
        for monitor,mode in zip(monitors, modes):
            self.modes[monitor] = mode
        self.filepath_monitors = filepath_monitors
        self.filepath_overall = filepath_overall
        self.start_epoch = start_epoch

        self.monitor_ops = {}
        self.best = {}
        self.prev_file_saved_monitors = ''
        self.prev_file_saved_overall = ''
        for monitor in self.monitors:
            if self.modes[monitor] == 'min':
                self.monitor_ops[monitor] = np.less
                self.best[monitor] = np.Inf
            elif self.modes[monitor] == 'max':
                self.monitor_ops[monitor] = np.greater
                self.best[monitor] = (-np.Inf)
            else:
                if 'acc' in monitor or monitor.startswith('fmeasure'):
                    self.monitor_ops[monitor] = np.greater
                    self.best[monitor] = -np.Inf
                else:
                    self.monitor_ops[monitor] = np.less
                    self.best[monitor] = np.Inf




    def on_epoch_end(self, epoch, logs=None):
        print('checkpointing model...')
        saved_monitors = False

        for monitor in self.monitors:
            if callable(monitor):
                current = monitor(logs)
                monitor_str = monitor.__name__
            else:
                try:
                    current = logs[monitor]
                    monitor_str = str(monitor)
                except:
                    print('unable to get monitor value \"%s\"' % monitor)
                    print('possible monitor names:')
                    for mname in logs.keys():
                        print(mname)

                    continue

            if self.monitor_ops[monitor](current, self.best[monitor]):
                self.best[monitor] = current

                if not saved_monitors:
                    fname = self.filepath_monitors.format(epoch=(epoch + self.start_epoch), monitor_name = monitor_str, monitor_value = current)
                    print("\nSaving monitor-defined checkpoint to : %s" % fname)

                    try:
                        self.model_to_save.save(fname, overwrite=True, include_optimizer=False)
                        saved_monitors = True
                    except:
                        print('unable to save new monitor-defined model checkpoint')

                    if os.path.isfile(self.prev_file_saved_monitors):
                        try:
                            os.remove(self.prev_file_saved_monitors)
                        except:
                            print('uneble to delete previously saved monitor-defined model checkpoint')

                    self.prev_file_saved_monitors = fname

        fname = self.filepath_overall.format(epoch=(epoch + self.start_epoch))
        print("\nSaving overall checkpoint to : %s" % fname)
        if saved_monitors:
            try:
                copyfile(self.prev_file_saved_monitors, fname)
                if os.path.isfile(self.prev_file_saved_overall):
                    try:
                        os.remove(self.prev_file_saved_overall)
                    except:
                        print('uneble to delete previously saved overall checkpoint')

                self.prev_file_saved_overall = fname
            except:
                print('uneble to copy monitor-defined model checkpoint to overall checkpoint')
        else:
            try:
                self.model_to_save.save(fname, overwrite=True, include_optimizer=False)
            except:
                print('unable to save new model weights')

            if os.path.isfile(self.prev_file_saved_overall):
                try:
                    os.remove(self.prev_file_saved_overall)
                except:
                    print('uneble to delete previously saved overall checkpoint')

            self.prev_file_saved_overall = fname
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


class CustomTensorBoard(TensorBoard):
  """Extends the TensorBoard callback to allow adding custom summaries.


  Arguments:
      user_defined_freq: frequency (in epochs) at which to compute summaries
          defined by the user by calling tf.summary in the model code. If set to
          0, user-defined summaries won't be computed. Validation data must be
          specified for summary visualization.
      kwargs: Passed to tf.keras.callbacks.TensorBoard.
  """


  def __init__(self, user_defined_freq=0, **kwargs):
    self.user_defined_freq = user_defined_freq
    super(CustomTensorBoard, self).__init__(**kwargs)


  def on_epoch_begin(self, epoch, logs=None):
    """Add user-def. op to Model eval_function callbacks, reset batch count."""

    # check if histogram summary should be run for this epoch
    if self.user_defined_freq and epoch % self.user_defined_freq == 0:
      self._epoch = epoch
      # pylint: disable=protected-access
      # add the user-defined summary ops if it should run this epoch
      self.model._make_eval_function()
      if self.merged not in self.model._eval_function.fetches:
        self.model._eval_function.fetches.append(self.merged)
        self.model._eval_function.fetch_callbacks[self.merged] = self._fetch_callback
      # pylint: enable=protected-access


    super(CustomTensorBoard, self).on_epoch_begin(epoch, logs=None)


  def on_epoch_end(self, epoch, logs=None):
    """Checks if summary ops should run next epoch, logs scalar summaries."""


    # pop the user-defined summary op after each epoch
    if self.user_defined_freq:
      # pylint: disable=protected-access
      if self.merged in self.model._eval_function.fetches:
        self.model._eval_function.fetches.remove(self.merged)
      if self.merged in self.model._eval_function.fetch_callbacks:
        self.model._eval_function.fetch_callbacks.pop(self.merged)
      # pylint: enable=protected-access


    super(CustomTensorBoard, self).on_epoch_end(epoch, logs=logs)
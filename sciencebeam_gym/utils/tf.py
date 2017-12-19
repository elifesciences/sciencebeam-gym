import tensorflow as tf

# pylint: disable=E0611
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors as tf_errors
# pylint: enable=E0611

def variable_scoped(name, fn):
  with tf.variable_scope(name):
    return fn()

def tf_print(x, message=None, **kwargs):
  return tf.Print(x, [x], message=message, **kwargs)

def FileIO(filename, mode):
  try:
    return file_io.FileIO(filename, mode)
  except tf_errors.InvalidArgumentError:
    if 'b' in mode:
      # older versions of TF don't support the 'b' flag as such
      return file_io.FileIO(filename, mode.replace('b', ''))
    else:
      raise

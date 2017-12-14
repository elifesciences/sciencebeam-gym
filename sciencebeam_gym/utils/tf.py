import tensorflow as tf

def variable_scoped(name, fn):
  with tf.variable_scope(name):
    return fn()

def tf_print(x, message=None, **kwargs):
  return tf.Print(x, [x], message=message, **kwargs)

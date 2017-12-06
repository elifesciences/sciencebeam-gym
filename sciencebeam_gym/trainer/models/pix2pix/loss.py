import tensorflow as tf

def l1_loss(labels, outputs):
  with tf.name_scope("l1_loss"):
    # abs(labels - outputs) => 0
    return tf.reduce_mean(tf.abs(labels - outputs))

def cross_entropy_loss(labels, logits):
  with tf.name_scope("cross_entropy"):
    return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        name='softmax_cross_entropy_with_logits'
      )
    )

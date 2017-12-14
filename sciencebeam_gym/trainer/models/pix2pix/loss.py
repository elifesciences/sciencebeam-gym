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
        labels=labels
      )
    )

def weighted_cross_entropy_loss(labels, logits, pos_weight, scalar=True):
  with tf.name_scope("weighted_cross_entropy"):
    softmax_loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits,
      labels=labels
    )
    # calculate weight per sample using the labels and weight per class
    weight_per_sample = tf.reduce_sum(
      tf.multiply(
        labels,
        pos_weight
      ),
      axis=-1
    )
    # weight each loss per sample
    value = tf.multiply(
      softmax_loss,
      weight_per_sample
    )
    return tf.reduce_mean(value) if scalar else value

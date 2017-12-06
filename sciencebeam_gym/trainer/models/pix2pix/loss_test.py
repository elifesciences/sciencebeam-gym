from six import raise_from

import tensorflow as tf
import numpy as np

from sciencebeam_gym.trainer.models.pix2pix.loss import (
  l1_loss,
  cross_entropy_loss
)

def assert_close(a, b):
  try:
    assert np.allclose([a], [b])
  except AssertionError as e:
    raise_from(AssertionError('expected %s to be close to %s' % (a, b)), e)

class TestL1Loss(object):
  def test_should_return_abs_diff_for_single_value(self):
    with tf.Graph().as_default():
      labels = tf.constant([0.9])
      outputs = tf.constant([0.1])
      loss = l1_loss(labels, outputs)
      with tf.Session() as session:
        assert_close(session.run([loss])[0], 0.8)

class TestCrossEntropyLoss(object):
  def test_should_return_zero_if_logits_are_matching_labels_with_neg_pos_value(self):
    with tf.Graph().as_default():
      labels = tf.constant([
        [[0.0, 1.0]]
      ])
      logits = tf.constant([
        [[-10.0, 10.0]]
      ])
      loss = cross_entropy_loss(labels, logits)
      with tf.Session() as session:
        assert_close(session.run([loss])[0], 0.0)

  def test_should_return_not_zero_if_logits_are_not_matching_labels(self):
    with tf.Graph().as_default():
      labels = tf.constant([
        [[0.0, 1.0]]
      ])
      logits = tf.constant([
        [[10.0, 10.0]]
      ])
      loss = cross_entropy_loss(labels, logits)
      with tf.Session() as session:
        assert session.run([loss])[0] > 0.5

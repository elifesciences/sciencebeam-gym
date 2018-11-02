import logging

import tensorflow as tf

from sciencebeam_utils.utils.num import (
    assert_close
)

from sciencebeam_gym.trainer.models.pix2pix.loss import (
    l1_loss,
    cross_entropy_loss,
    weighted_cross_entropy_loss
)


def get_logger():
    return logging.getLogger(__name__)


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


class TestWeightedCrossEntropyLoss(object):
    def test_should_return_zero_if_logits_are_matching_labels_with_neg_pos_value(self):
        with tf.Graph().as_default():
            labels = tf.constant([
                [[0.0, 1.0]]
            ])
            logits = tf.constant([
                [[-10.0, 10.0]]
            ])
            pos_weight = tf.constant([
                1.0
            ])
            loss = weighted_cross_entropy_loss(labels, logits, pos_weight)
            with tf.Session() as session:
                assert_close(session.run([loss])[0], 0.0, atol=0.0001)

    def test_should_return_not_zero_if_logits_are_not_matching_labels(self):
        with tf.Graph().as_default():
            labels = tf.constant([
                [[0.0, 1.0]]
            ])
            logits = tf.constant([
                [[10.0, 10.0]]
            ])
            pos_weight = tf.constant([
                1.0
            ])
            loss = weighted_cross_entropy_loss(labels, logits, pos_weight)
            with tf.Session() as session:
                assert session.run([loss])[0] > 0.5

    def test_should_return_higher_loss_for_value_with_greater_weight(self):
        with tf.Graph().as_default():
            labels = tf.constant([
                [[0.0, 1.0]]
            ])
            logits = tf.constant([
                [[10.0, 10.0]]
            ])
            pos_weight_1 = tf.constant([
                0.5
            ])
            pos_weight_2 = tf.constant([
                1.0
            ])
            loss_1 = weighted_cross_entropy_loss(labels, logits, pos_weight_1)
            loss_2 = weighted_cross_entropy_loss(labels, logits, pos_weight_2)
            with tf.Session() as session:
                loss_1_value, loss_2_value = session.run([loss_1, loss_2])
                assert loss_1_value < loss_2_value

    def test_should_support_batch_example_pos_weights(self):
        batch_size = 3
        with tf.Graph().as_default():
            labels = tf.constant([[0.0, 1.0]] * batch_size)
            logits = tf.constant([[10.0, 10.0]] * batch_size)
            pos_weight_1 = tf.constant([
                [0.5, 0.5],
                [1.0, 1.0],
                [1.0, 1.0]
            ])
            pos_weight_2 = tf.constant([
                [1.0, 1.0],
                [0.5, 0.5],
                [1.0, 1.0]
            ])
            loss_1 = weighted_cross_entropy_loss(
                labels, logits, pos_weight_1, scalar=False
            )
            loss_2 = weighted_cross_entropy_loss(
                labels, logits, pos_weight_2, scalar=False
            )
            with tf.Session() as session:
                get_logger().debug('labels=\n%s', labels.eval())
                get_logger().debug('logits=\n%s', logits.eval())
                loss_1_value, loss_2_value = session.run([loss_1, loss_2])
                get_logger().debug(
                    '\nloss_1_value=\n%s\nloss_2_value=\n%s',
                    loss_1_value, loss_2_value
                )
                assert loss_1_value[0] < loss_2_value[0]
                assert loss_1_value[1] > loss_2_value[1]
                assert loss_1_value[2] == loss_2_value[2]

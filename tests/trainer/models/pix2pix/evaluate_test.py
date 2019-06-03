from __future__ import absolute_import
from __future__ import division

import pytest

import tensorflow as tf
import numpy as np

from sciencebeam_utils.utils.num import (
    assert_close
)

from sciencebeam_gym.trainer.models.pix2pix.evaluate import (
    evaluate_predictions
)


@pytest.mark.slow
def test_evaluate_predictions():
    n_classes = 4
    predictions = tf.constant(np.array([0, 1, 1, 2, 3, 3]))
    labels = tf.constant(np.array([0, 1, 2, 3, 3, 3]))

    evaluation_tensors = evaluate_predictions(
        labels=labels,
        predictions=predictions,
        n_classes=n_classes
    )
    with tf.Session() as session:
        assert np.array_equal(session.run(evaluation_tensors.confusion_matrix), np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2]
        ]))
        assert np.array_equal(session.run(evaluation_tensors.tp), np.array([1, 1, 0, 2]))
        assert np.array_equal(session.run(evaluation_tensors.fp), np.array([0, 1, 1, 0]))
        assert np.array_equal(session.run(evaluation_tensors.fn), np.array([0, 0, 1, 1]))
        expected_micro_precision = 4.0 / (4 + 2)
        expected_micro_recall = 4.0 / (4 + 2)
        expected_micro_f1 = (
            2 * expected_micro_precision * expected_micro_recall /
            (expected_micro_precision + expected_micro_recall)
        )
        assert_close(
            session.run(evaluation_tensors.micro_precision),
            expected_micro_precision
        )
        assert_close(
            session.run(evaluation_tensors.micro_recall),
            expected_micro_recall
        )
        assert_close(
            session.run(evaluation_tensors.micro_f1),
            expected_micro_f1
        )

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

from sciencebeam_gym.trainer.models.pix2pix.tf_utils import (
  find_nearest_centroid
)

def test_find_nearest_centroid():
  colors = tf.constant([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
  outputs = tf.constant([[[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.9]]])
  nearest_color = find_nearest_centroid(outputs, colors)

  with tf.Session() as session:
    assert np.allclose(
      session.run(nearest_color),
      [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]
    )

def test_find_nearest_centroid1():
  colors = tf.constant([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
  outputs = tf.constant([
    [
      [[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.9]],
      [[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.9]]
    ],
    [
      [[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.9]],
      [[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.9]]
    ]
  ])
  nearest_color = find_nearest_centroid(outputs, colors)

  with tf.Session() as session:
    assert np.allclose(
      session.run(nearest_color),
      [
        [
          [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
          [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
        ],
        [
          [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
          [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
        ]
      ]
    )

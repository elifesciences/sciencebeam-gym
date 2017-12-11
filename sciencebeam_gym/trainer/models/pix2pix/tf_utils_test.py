from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

from sciencebeam_gym.utils.num import (
  assert_all_close
)

from sciencebeam_gym.trainer.models.pix2pix.tf_utils import (
  find_nearest_centroid,
  blank_other_channels
)

def test_find_nearest_centroid():
  colors = tf.constant([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
  outputs = tf.constant([[[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.9]]])
  nearest_color = find_nearest_centroid(outputs, colors)

  with tf.Session() as session:
    assert_all_close(
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
    assert_all_close(
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

def test_blank_other_channels():
  tensor = tf.constant([
    [
      [5, 5, 5, 5],
      [6, 6, 6, 6],
      [7, 7, 7, 7],
      [8, 8, 8, 8]
    ],
    [
      [5, 5, 5, 5],
      [6, 6, 6, 6],
      [7, 7, 7, 7],
      [8, 8, 8, 8]
    ]
  ])
  padded = blank_other_channels(
    tensor, 1
  )
  with tf.Session() as session:
    assert_all_close(
      session.run(padded),
      [
        [
          [0, 5, 0, 0],
          [0, 6, 0, 0],
          [0, 7, 0, 0],
          [0, 8, 0, 0]
        ],
        [
          [0, 5, 0, 0],
          [0, 6, 0, 0],
          [0, 7, 0, 0],
          [0, 8, 0, 0]
        ]
      ]
    )

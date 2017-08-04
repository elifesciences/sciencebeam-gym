from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

def pairwise_squared_distance_with(predictions, centroids):
  centroid_count = centroids.shape[0]
  return tf.stack([
    tf.reduce_sum(
      tf.square(predictions - centroids[i]),
      axis=-1
    )
    for i in range(centroid_count)
  ], axis=-1)

def find_nearest_centroid_indices(predictions, centroids):
  return tf.argmin(
    pairwise_squared_distance_with(predictions, centroids),
    axis=-1
  )

def find_nearest_centroid(predictions, centroids):
  return tf.gather(
    params=centroids,
    indices=find_nearest_centroid_indices(predictions, centroids)
  )

def get_channel_slice(tensor, channel_index):
  rank = len(tensor.shape)
  return tf.slice(
    tensor,
    begin=[0] * (rank - 1) + [channel_index],
    size=[-1] * (rank - 1) + [1]
  )

def blank_other_channels(tensor, keep_index):
  tensor_shape = tensor.shape
  n_channels = int(tensor_shape[-1])
  rank = len(tensor_shape)
  tensor_slice = get_channel_slice(
    tensor,
    keep_index
  )
  paddings = tf.constant(
    [[0, 0]] * (rank - 1) +
    [[keep_index, n_channels - keep_index - 1]]
  )
  padded = tf.pad(
    tensor_slice, paddings, "CONSTANT"
  )
  return padded

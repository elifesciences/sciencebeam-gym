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

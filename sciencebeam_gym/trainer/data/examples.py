import logging
from functools import partial

import tensorflow as tf
from tensorflow.python.lib.io import file_io # pylint: disable=E0611

from sciencebeam_utils.utils.collection import (
  extend_dict
)

from sciencebeam_gym.model_utils.channels import (
  calculate_color_masks
)

try:
  # TensorFlow 1.4+
  tf_data = tf.data
except AttributeError:
  tf_data = tf.contrib.data

Dataset = tf_data.Dataset
TFRecordDataset = tf_data.TFRecordDataset

DEFAULT_FEATURE_MAP = {
  'input_uri': tf.FixedLenFeature(
    shape=[], dtype=tf.string, default_value=['']
  ),
  'annotation_uri': tf.FixedLenFeature(
    shape=[], dtype=tf.string, default_value=['']
  ),
  'input_image': tf.FixedLenFeature(
    shape=[], dtype=tf.string
  ),
  'annotation_image': tf.FixedLenFeature(
    shape=[], dtype=tf.string
  )
}

PAGE_NO_FEATURE = {
  'page_no': tf.FixedLenFeature(
    shape=[], dtype=tf.int64
  )
}

def get_logger():
  return logging.getLogger(__name__)

def get_matching_files(paths):
  files = []
  for e in paths:
    for path in e.split(','):
      files.extend(file_io.get_matching_files(path))
  return files

def parse_example(example, feature_map=None):
  if feature_map is None:
    feature_map = DEFAULT_FEATURE_MAP
  get_logger().info('example: %s', example)
  return tf.parse_single_example(example, features=feature_map)

# Workaround for Tensorflow 1.2 not supporting dicts
class MapKeysTracker(object):
  def __init__(self):
    self.keys = None

  def wrap(self, fn):
    def wrapper(x):
      x = fn(x)
      if self.keys is not None:
        get_logger().warn('keys already set: %s', self.keys)
      self.keys = sorted(x.keys())
      return [x[k] for k in self.keys]
    return wrapper

  def unwrap(self, result):
    return {k: v for k, v in zip(self.keys, result)}

def page_no_is_within(page_no, page_range):
  get_logger().debug('page_no: %s, page_range: %s', page_no, page_range)
  return tf.logical_and(page_no >= page_range[0], page_no <= page_range[1])

def image_contains_any_of_the_colors(image, colors):
  decoded_image = tf.image.decode_png(image, channels=3)
  color_masks = calculate_color_masks(decoded_image, colors)
  return tf.reduce_any([
    tf.reduce_any(color_mask >= 0.5)
    for color_mask in color_masks
  ])

def read_examples(
  filenames,
  shuffle,
  num_epochs=None,
  page_range=None,
  channel_colors=None):

  # Convert num_epochs == 0 -> num_epochs is None, if necessary
  num_epochs = num_epochs or None

  feature_map = DEFAULT_FEATURE_MAP
  if page_range is not None:
    feature_map = extend_dict(feature_map, PAGE_NO_FEATURE)

  map_keys_tracker = MapKeysTracker()

  dataset = TFRecordDataset(filenames, compression_type='GZIP')
  dataset = dataset.map(map_keys_tracker.wrap(
    partial(parse_example, feature_map=feature_map)
  ))
  if page_range is not None:
    dataset = dataset.filter(lambda *x: page_no_is_within(
      map_keys_tracker.unwrap(x)['page_no'],
      page_range
    ))
  if channel_colors is not None:
    dataset = dataset.filter(lambda *x: image_contains_any_of_the_colors(
      map_keys_tracker.unwrap(x)['annotation_image'],
      channel_colors
    ))
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(num_epochs)

  return map_keys_tracker.unwrap(
    dataset.make_one_shot_iterator().get_next()
  )

import logging
from functools import partial

import tensorflow as tf
from tensorflow.python.lib.io import file_io # pylint: disable=E0611

from sciencebeam_gym.utils.collection import (
  extend_dict
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

def page_no_is_within(page_no, page_range):
  get_logger().debug('page_no: %s, page_range: %s', page_no, page_range)
  return tf.logical_and(page_no >= page_range[0], page_no <= page_range[1])

def read_examples(filenames, shuffle, num_epochs=None, page_range=None):
  # Convert num_epochs == 0 -> num_epochs is None, if necessary
  num_epochs = num_epochs or None

  feature_map = DEFAULT_FEATURE_MAP
  if page_range is not None:
    feature_map = extend_dict(feature_map, PAGE_NO_FEATURE)

  dataset = TFRecordDataset(filenames, compression_type='GZIP')
  dataset = dataset.map(partial(parse_example, feature_map=feature_map))
  if page_range is not None:
    dataset = dataset.filter(lambda x: page_no_is_within(x['page_no'], page_range))
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(num_epochs)

  return dataset.make_one_shot_iterator().get_next()

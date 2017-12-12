import logging

import tensorflow as tf
from tensorflow.python.lib.io import file_io # pylint: disable=E0611

try:
  # TensorFlow 1.4+
  tf_data = tf.data
except AttributeError:
  tf_data = tf.contrib.data

Dataset = tf_data.Dataset
TFRecordDataset = tf_data.TFRecordDataset

def get_logger():
  return logging.getLogger(__name__)

def get_matching_files(paths):
  files = []
  for e in paths:
    for path in e.split(','):
      files.extend(file_io.get_matching_files(path))
  return files

def parse_example(example):
  feature_map = {
    'input_uri':
      tf.FixedLenFeature(
        shape=[], dtype=tf.string, default_value=['']
      ),
    'annotation_uri':
      tf.FixedLenFeature(
        shape=[], dtype=tf.string, default_value=['']
      ),
    'input_image':
      tf.FixedLenFeature(
        shape=[], dtype=tf.string
      ),
    'annotation_image':
      tf.FixedLenFeature(
        shape=[], dtype=tf.string
      )
  }
  get_logger().info('example: %s', example)
  return tf.parse_single_example(example, features=feature_map)

def read_examples(filenames, shuffle, num_epochs=None):
  # Convert num_epochs == 0 -> num_epochs is None, if necessary
  num_epochs = num_epochs or None

  dataset = TFRecordDataset(filenames, compression_type='GZIP')
  dataset = dataset.map(parse_example)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(num_epochs)

  return dataset.make_one_shot_iterator().get_next()

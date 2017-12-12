import logging
from mock import patch

import pytest

import tensorflow as tf

from sciencebeam_gym.utils.collection import (
  extend_dict
)

from sciencebeam_gym.utils.tfrecord import (
  dict_to_example
)

import sciencebeam_gym.trainer.data.examples as examples_module
from sciencebeam_gym.trainer.data.examples import (
  read_examples,
  tf_data
)

DATA_PATH = '.temp/data/*.tfrecord'


EXAMPLE_PROPS_1 = {
  'input_uri': 'input.png',
  'input_image': b'input image',
  'annotation_uri': 'annotation.png',
  'annotation_image': b'annotation image'
}

RECORD_1 = dict_to_example(EXAMPLE_PROPS_1).SerializeToString()

def get_logger():
  return logging.getLogger(__name__)

def setup_module():
  logging.basicConfig(level='DEBUG')

def list_dataset(data, dtype):
  data = tf.constant(data, dtype=dtype)
  return tf_data.Dataset.from_tensor_slices(data)

def fetch_examples(session, examples_tensor, max_examples=1000):
  try:
    for _ in range(max_examples):
      yield session.run([examples_tensor])[0]
  except tf.errors.OutOfRangeError:
    get_logger().debug('end of dataset')

@pytest.mark.slow
class TestReadExamples(object):
  def test_should_read_single_example(self):
    with patch.object(examples_module, 'TFRecordDataset') as TFRecordDataset:
      with tf.Graph().as_default():
        TFRecordDataset.return_value = list_dataset([RECORD_1], tf.string)
        examples = read_examples(DATA_PATH, shuffle=False)
        TFRecordDataset.assert_called_with(DATA_PATH, compression_type='GZIP')
        with tf.Session() as session:
          next_example = session.run([examples])[0]
          get_logger().info('next_example: %s', next_example)
          assert next_example == EXAMPLE_PROPS_1

  def test_should_filter_by_page_no(self):
    with patch.object(examples_module, 'TFRecordDataset') as TFRecordDataset:
      with tf.Graph().as_default():
        TFRecordDataset.return_value = list_dataset([
          dict_to_example(extend_dict(EXAMPLE_PROPS_1, page_no=page_no)).SerializeToString()
          for page_no in [1, 2, 3, 4]
        ], tf.string)
        examples = read_examples(DATA_PATH, shuffle=False, num_epochs=1, page_range=(2, 3))
        TFRecordDataset.assert_called_with(DATA_PATH, compression_type='GZIP')
        with tf.Session() as session:
          assert [x['page_no'] for x in fetch_examples(session, examples)] == [2, 3]

from mock import patch

import tensorflow as tf

import sciencebeam_gym.trainer.data.examples as examples_module
from sciencebeam_gym.trainer.data.examples import (
  read_examples,
  tf_data
)

DATA_PATH = '.temp/data/*.tfrecord'

RECORD_1 = b'record 1'

class TestReadExamples(object):
  def test_should_read_single_example(self):
    with patch.object(examples_module, 'TFRecordDataset') as TFRecordDataset:
      with tf.Graph().as_default():
        TFRecordDataset.return_value = tf_data.Dataset.from_generator(
          lambda: [RECORD_1],
          tf.string
        )
        examples = read_examples(DATA_PATH, shuffle=False)
        TFRecordDataset.assert_called_with(DATA_PATH, compression_type='GZIP')
        with tf.Session() as session:
          next_example = session.run([examples])[0]
          assert next_example == RECORD_1

from __future__ import absolute_import

import tensorflow as tf

def iter_examples_to_dict_list(examples):
  for example in examples:
    result = tf.train.Example.FromString(example)
    yield {
     key: result.features.feature.get(key).bytes_list.value[0]
     for key in result.features.feature.keys()
    }

def iter_read_tfrecord_file_as_dict_list(filename):
  options = None
  if filename.endswith('.gz'):
    options = tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP
    )
  examples = tf.python_io.tf_record_iterator(filename, options=options)
  return iter_examples_to_dict_list(examples)

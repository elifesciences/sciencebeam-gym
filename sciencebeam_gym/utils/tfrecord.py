from __future__ import absolute_import

from six import iteritems, raise_from, text_type

import tensorflow as tf

def iter_examples_to_dict_list(examples, keys=None):
  for example in examples:
    result = tf.train.Example.FromString(example)
    yield {
     key: result.features.feature.get(key).bytes_list.value[0]
     for key in result.features.feature.keys()
     if keys is None or key in keys
    }

def iter_read_tfrecord_file_as_dict_list(filename, keys=None):
  options = None
  if filename.endswith('.gz'):
    options = tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP
    )
  examples = tf.python_io.tf_record_iterator(filename, options=options)
  return iter_examples_to_dict_list(examples, keys=keys)

def _bytes_feature(value, name):
  try:
    if isinstance(value, text_type):
      value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
  except TypeError as e:
    raise_from(TypeError('failed to convert %s due to %s' % (name, e)), e)

def dict_to_example(props):
  return tf.train.Example(features=tf.train.Features(feature={
    k: _bytes_feature([v], name=k)
    for k, v in iteritems(props)
  }))

def write_examples_to_tfrecord(tfrecord_filename, examples):
  with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

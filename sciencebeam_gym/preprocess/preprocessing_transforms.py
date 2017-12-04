from six import iteritems, raise_from, text_type

import apache_beam as beam

try:
  import tensorflow as tf
except ImportError:
  # Make tensorflow optional
  tf = None


def _bytes_feature(value, name):
  try:
    if isinstance(value, text_type):
      value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
  except TypeError as e:
    raise_from(TypeError('failed to convert %s due to %s' % (name, e)), e)

class WritePropsToTFRecord(beam.PTransform):
  def __init__(self, file_path, extract_props):
    super(WritePropsToTFRecord, self).__init__()
    self.file_path = file_path
    self.extract_props = extract_props
    if tf is None:
      raise RuntimeError('TensorFlow required for this transform')

  def expand(self, pcoll):
    return (
      pcoll |
      'ConvertToTfExamples' >> beam.FlatMap(lambda v: (
        tf.train.Example(features=tf.train.Features(feature={
          k: _bytes_feature([v], name=k)
          for k, v in iteritems(props)
        }))
        for props in self.extract_props(v)
      )) |
      'SerializeToString' >> beam.Map(lambda x: x.SerializeToString()) |
      'SaveToTfRecords' >> beam.io.WriteToTFRecord(
        self.file_path,
        file_name_suffix='.tfrecord.gz'
      )
    )

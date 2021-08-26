import logging

import apache_beam as beam

try:
    import tensorflow as tf

    from sciencebeam_gym.utils.tfrecord import (
        dict_to_example
    )
except ImportError:
    # Make tensorflow optional
    tf = None


LOGGER = logging.getLogger(__name__)


class WritePropsToTFRecord(beam.PTransform):
    def __init__(self, file_path, extract_props, file_name_suffix='.tfrecord.gz'):
        super().__init__()
        self.file_path = file_path
        self.extract_props = extract_props
        self.file_name_suffix = file_name_suffix
        if tf is None:
            raise RuntimeError('TensorFlow required for this transform')
        LOGGER.debug('tfrecords output file: %r', self.file_path + self.file_name_suffix)

    def expand(self, input_or_inputs):  # pylint: disable=W0221
        return (
            input_or_inputs |
            'ConvertToTfExamples' >> beam.FlatMap(lambda v: (
                dict_to_example(props)
                for props in self.extract_props(v)
            )) |
            'SerializeToString' >> beam.Map(lambda x: x.SerializeToString()) |
            'SaveToTfRecords' >> beam.io.WriteToTFRecord(
                self.file_path,
                file_name_suffix=self.file_name_suffix
            )
        )

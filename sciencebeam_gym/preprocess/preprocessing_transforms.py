import apache_beam as beam

try:
    import tensorflow as tf

    from sciencebeam_gym.utils.tfrecord import (
        dict_to_example
    )
except ImportError:
    # Make tensorflow optional
    tf = None


class WritePropsToTFRecord(beam.PTransform):
    def __init__(self, file_path, extract_props):
        super(WritePropsToTFRecord, self).__init__()
        self.file_path = file_path
        self.extract_props = extract_props
        if tf is None:
            raise RuntimeError('TensorFlow required for this transform')

    def expand(self, pcoll):  # pylint: disable=W0221
        return (
            pcoll |
            'ConvertToTfExamples' >> beam.FlatMap(lambda v: (
                dict_to_example(props)
                for props in self.extract_props(v)
            )) |
            'SerializeToString' >> beam.Map(lambda x: x.SerializeToString()) |
            'SaveToTfRecords' >> beam.io.WriteToTFRecord(
                self.file_path,
                file_name_suffix='.tfrecord.gz'
            )
        )

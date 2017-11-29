import apache_beam as beam
from apache_beam.io.filesystems import FileSystems

from sciencebeam_gym.beam_utils.io import (
  find_matching_filenames
)

from sciencebeam_gym.utils.tfrecord import (
  iter_read_tfrecord_file_as_dict_list
)

from sciencebeam_gym.beam_utils.testing import (
  BeamTest,
  TestPipeline
)

from sciencebeam_gym.preprocess.preprocessing_transforms import (
  WritePropsToTFRecord
)

TFRECORDS_PATH = '.temp/test-data'

KEY_1 = b'key1'
KEY_2 = b'key2'

VALUE_1 = b'value 1'
VALUE_2 = b'value 2'

class TestWritePropsToTFRecord(BeamTest):
  def test_should_write_single_entry(self):
    dict_list = [{KEY_1: VALUE_1}]
    with TestPipeline() as p:
      _ = (
        p |
        beam.Create(dict_list) |
        WritePropsToTFRecord(TFRECORDS_PATH, lambda x: [x])
      )
    filenames = list(find_matching_filenames(TFRECORDS_PATH + '*'))
    assert len(filenames) == 1
    records = list(iter_read_tfrecord_file_as_dict_list(filenames[0]))
    assert records == dict_list
    FileSystems.delete(filenames)

  def test_should_write_multiple_entries(self):
    dict_list = [
      {KEY_1: VALUE_1},
      {KEY_2: VALUE_2}
    ]
    with TestPipeline() as p:
      _ = (
        p |
        beam.Create(dict_list) |
        WritePropsToTFRecord(TFRECORDS_PATH, lambda x: [x])
      )
    filenames = list(find_matching_filenames(TFRECORDS_PATH + '*'))
    assert len(filenames) == 1
    records = list(iter_read_tfrecord_file_as_dict_list(filenames[0]))
    assert records == dict_list
    FileSystems.delete(filenames)

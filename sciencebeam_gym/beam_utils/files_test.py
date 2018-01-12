from mock import patch

import apache_beam as beam
from apache_beam.testing.util import (
  assert_that,
  equal_to
)

from sciencebeam_gym.beam_utils.testing import (
  BeamTest,
  TestPipeline
)

import sciencebeam_gym.beam_utils.files as files_module
from sciencebeam_gym.beam_utils.files import (
  ReadFileList,
  DeferredReadFileList,
  FindFiles,
  DeferredFindFiles
)

FILE_1 = 'file1.pdf'
FILE_2 = 'file2.pdf'

FILE_LIST_PATH = 'file-list.lst'
COLUMN = 'url'
LIMIT = 10

class TestReadFileList(BeamTest):
  def test_should_use_load_file_list(self):
    with patch.object(files_module, 'load_file_list') as load_file_list:
      load_file_list.return_value = [FILE_1, FILE_2]
      with TestPipeline() as p:
        result = p | ReadFileList(FILE_LIST_PATH, column=COLUMN, limit=LIMIT)
        assert_that(result, equal_to([FILE_1, FILE_2]))
      load_file_list.assert_called_with(FILE_LIST_PATH, column=COLUMN, limit=LIMIT)

class TestDeferredReadFileList(BeamTest):
  def test_should_use_read_dict_csv(self):
    with patch.object(files_module, 'ReadDictCsv') as ReadDictCsv:
      ReadDictCsv.return_value = beam.Create([{COLUMN: FILE_1}, {COLUMN: FILE_2}])
      with TestPipeline() as p:
        result = p | DeferredReadFileList(FILE_LIST_PATH, column=COLUMN, limit=LIMIT)
        assert_that(result, equal_to([FILE_1, FILE_2]))
      ReadDictCsv.assert_called_with(FILE_LIST_PATH, limit=LIMIT)

class TestFindFiles(BeamTest):
  def test_should_use_find_matching_filenames(self):
    with patch.object(files_module, 'find_matching_filenames') as find_matching_filenames:
      find_matching_filenames.return_value = [FILE_1, FILE_2]
      with TestPipeline() as p:
        result = p | FindFiles(FILE_LIST_PATH, limit=LIMIT)
        assert_that(result, equal_to([FILE_1, FILE_2]))
      find_matching_filenames.assert_called_with(FILE_LIST_PATH)

  def test_should_apply_limit(self):
    with patch.object(files_module, 'find_matching_filenames') as find_matching_filenames:
      find_matching_filenames.return_value = [FILE_1, FILE_2]
      with TestPipeline() as p:
        result = p | FindFiles(FILE_LIST_PATH, limit=1)
        assert_that(result, equal_to([FILE_1]))

class TestDeferredFindFiles(BeamTest):
  def test_should_use_find_matching_filenames(self):
    with patch.object(files_module, 'find_matching_filenames') as find_matching_filenames:
      find_matching_filenames.return_value = [FILE_1, FILE_2]
      with TestPipeline() as p:
        result = p | DeferredFindFiles(FILE_LIST_PATH, limit=LIMIT)
        assert_that(result, equal_to([FILE_1, FILE_2]))
      find_matching_filenames.assert_called_with(FILE_LIST_PATH)

  def test_should_apply_limit(self):
    with patch.object(files_module, 'find_matching_filenames') as find_matching_filenames:
      find_matching_filenames.return_value = [FILE_1, FILE_2]
      with TestPipeline() as p:
        result = p | DeferredFindFiles(FILE_LIST_PATH, limit=1)
        assert_that(result, equal_to([FILE_1]))

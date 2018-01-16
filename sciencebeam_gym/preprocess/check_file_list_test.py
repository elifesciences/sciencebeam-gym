from mock import patch

import pytest

import sciencebeam_gym.preprocess.check_file_list as check_file_list_module
from sciencebeam_gym.preprocess.check_file_list import (
  map_file_list_to_file_exists,
  format_file_exists_results,
  check_files_and_report_result
)

FILE_1 = 'file1'
FILE_2 = 'file2'

class TestMapFileListToFileExists(object):
  def test_should_return_single_file_exists(self):
    m = check_file_list_module
    with patch.object(m, 'FileSystems') as FileSystems:
      assert map_file_list_to_file_exists(
        [FILE_1]
      ) == [FileSystems.exists.return_value]
      FileSystems.exists.assert_called_with(FILE_1)

class TestFormatFileExistsResults(object):
  def test_should_format_no_files(self):
    assert (
      format_file_exists_results([]) ==
      'empty file list'
    )

  def test_should_format_all_files_exist(self):
    assert (
      format_file_exists_results([True, True]) ==
      'files exist: 2 (100%), files missing: 0 (0%)'
    )

  def test_should_format_files_partially_exist(self):
    assert (
      format_file_exists_results([True, False]) ==
      'files exist: 1 (50%), files missing: 1 (50%)'
    )

class TestCheckFileListAndReportResults(object):
  def test_should_pass_file_list_to_format(self):
    m = check_file_list_module
    with patch.object(m, 'map_file_list_to_file_exists') as map_file_list_to_file_exists_mock:
      with patch.object(m, 'format_file_exists_results') as format_file_exists_results_mock:
        map_file_list_to_file_exists_mock.return_value = [True, True]
        check_files_and_report_result([FILE_1, FILE_2])
        map_file_list_to_file_exists_mock.assert_called_with([FILE_1, FILE_2])
        format_file_exists_results_mock.assert_called_with([True, True])

  def test_should_raise_error_if_none_of_the_files_were_found(self):
    m = check_file_list_module
    with patch.object(m, 'map_file_list_to_file_exists') as map_file_list_to_file_exists_mock:
      with pytest.raises(AssertionError):
        map_file_list_to_file_exists_mock.return_value = [False, False]
        check_files_and_report_result([FILE_1, FILE_2])

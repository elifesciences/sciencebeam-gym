import os
from mock import patch, ANY

import pytest

import sciencebeam_gym.preprocess.get_output_files as get_output_files
from sciencebeam_gym.preprocess.get_output_files import (
  get_output_file_list,
  run,
  parse_args,
  main
)

SOME_ARGV = [
  '--source-file-list=source.csv',
  '--output-file-list=output.csv',
  '--limit=10'
]

BASE_SOURCE_PATH = '/source'

FILE_1 = BASE_SOURCE_PATH + '/file1'
FILE_2 = BASE_SOURCE_PATH + '/file2'


@pytest.fixture(name='load_file_list_mock')
def _load_file_list():
  with patch.object(get_output_files, 'load_file_list') as m:
    m.return_value = [FILE_1, FILE_2]
    yield m

@pytest.fixture(name='get_output_file_list_mock')
def _get_output_file_list():
  with patch.object(get_output_files, 'get_output_file_list') as m:
    yield m

@pytest.fixture(name='save_file_list_mock')
def _save_file_list():
  with patch.object(get_output_files, 'save_file_list') as m:
    yield m

@pytest.fixture(name='check_files_and_report_result_mock')
def _check_files_and_report_result():
  with patch.object(get_output_files, 'check_files_and_report_result') as m:
    yield m

@pytest.fixture(name='to_relative_file_list_mock')
def _to_relative_file_list():
  with patch.object(get_output_files, 'to_relative_file_list') as m:
    yield m

class TestGetOutputFileList(object):
  def test_should_return_output_file_with_path_and_change_ext(self):
    assert get_output_file_list(
      ['/source/path/file.pdf'],
      '/source',
      '/output',
      '.xml'
    ) == ['/output/path/file.xml']

@pytest.mark.usefixtures(
  "load_file_list_mock", "get_output_file_list_mock", "save_file_list_mock",
  "to_relative_file_list_mock"
)
class TestRun(object):
  def test_should_pass_around_parameters(
    self,
    load_file_list_mock,
    get_output_file_list_mock,
    save_file_list_mock):

    load_file_list_mock.return_value = [FILE_1, FILE_2]
    opt = parse_args(SOME_ARGV)
    run(opt)
    load_file_list_mock.assert_called_with(
      opt.source_file_list,
      column=opt.source_file_column,
      limit=opt.limit
    )
    get_output_file_list_mock.assert_called_with(
      load_file_list_mock.return_value,
      BASE_SOURCE_PATH,
      opt.output_base_path,
      opt.output_file_suffix
    )
    save_file_list_mock.assert_called_with(
      opt.output_file_list,
      get_output_file_list_mock.return_value,
      column=opt.source_file_column
    )

  def test_should_make_file_list_absolute_if_it_is_relative(
    self,
    load_file_list_mock):

    opt = parse_args(SOME_ARGV)
    opt.source_base_path = BASE_SOURCE_PATH
    opt.source_file_list = 'source.tsv'
    run(opt)
    load_file_list_mock.assert_called_with(
      os.path.join(opt.source_base_path, opt.source_file_list),
      column=opt.source_file_column,
      limit=opt.limit
    )

  def test_should_raise_error_if_source_path_is_invalid(self):
    opt = parse_args(SOME_ARGV)
    opt.source_base_path = '/other/path'
    with pytest.raises(AssertionError):
      run(opt)

  def test_should_use_passed_in_source_path_if_valid(
    self,
    get_output_file_list_mock,
    load_file_list_mock):

    opt = parse_args(SOME_ARGV)
    opt.source_base_path = '/base'
    load_file_list_mock.return_value = ['/base/source/file1', '/base/source/file2']
    run(opt)
    get_output_file_list_mock.assert_called_with(
      ANY,
      opt.source_base_path,
      ANY,
      ANY
    )

  def test_should_check_file_list_if_enabled(
    self,
    get_output_file_list_mock,
    check_files_and_report_result_mock):

    opt = parse_args(SOME_ARGV)
    opt.check = True
    run(opt)
    check_files_and_report_result_mock.assert_called_with(
      get_output_file_list_mock.return_value
    )

  def test_should_limit_files_to_check(
    self,
    load_file_list_mock,
    get_output_file_list_mock,
    check_files_and_report_result_mock):

    opt = parse_args(SOME_ARGV)
    opt.check = True
    opt.check_limit = 1
    load_file_list_mock.return_value = [FILE_1, FILE_2]
    run(opt)
    check_files_and_report_result_mock.assert_called_with(
      get_output_file_list_mock.return_value[:opt.check_limit]
    )

  def test_should_save_relative_paths_if_enabled(
    self,
    get_output_file_list_mock,
    to_relative_file_list_mock,
    save_file_list_mock):

    opt = parse_args(SOME_ARGV)
    opt.use_relative_paths = True
    run(opt)
    to_relative_file_list_mock.assert_called_with(
      opt.output_base_path,
      get_output_file_list_mock.return_value,
    )
    save_file_list_mock.assert_called_with(
      opt.output_file_list,
      to_relative_file_list_mock.return_value,
      column=opt.source_file_column
    )

class TestMain(object):
  def test_should_parse_args_and_call_run(self):
    m = get_output_files
    with patch.object(m, 'parse_args') as parse_args_mock:
      with patch.object(m, 'run') as run_mock:
        main(SOME_ARGV)
        parse_args_mock.assert_called_with(SOME_ARGV)
        run_mock.assert_called_with(parse_args_mock.return_value)

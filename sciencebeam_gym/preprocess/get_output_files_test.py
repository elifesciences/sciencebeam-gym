from mock import patch

import sciencebeam_gym.preprocess.get_output_files as get_output_files
from sciencebeam_gym.preprocess.get_output_files import (
  get_output_file_list,
  run,
  parse_args,
  main
)

SOME_ARGV = [
  '--source-file-list=source.csv',
  '--source-base-path=/source',
  '--output-file-list=output.csv',
  '--limit=10'
]

class TestGetOutputFileList(object):
  def test_should_return_output_file_with_path_and_change_ext(self):
    assert get_output_file_list(
      ['/source/path/file.pdf'],
      '/source',
      '/output',
      '.xml'
    ) == ['/output/path/file.xml']

class TestRun(object):
  def test_should_pass_around_parameters(self):
    m = get_output_files
    opt = parse_args(SOME_ARGV)
    with patch.object(m, 'load_file_list') as load_file_list:
      with patch.object(m, 'get_output_file_list') as get_output_file_list_mock:
        with patch.object(m, 'save_file_list') as save_file_list:
          run(opt)
          load_file_list.assert_called_with(
            opt.source_file_list,
            column=opt.source_file_column,
            limit=opt.limit
          )
          get_output_file_list_mock.assert_called_with(
            load_file_list.return_value,
            opt.source_base_path,
            opt.output_base_path,
            opt.output_file_suffix
          )
          save_file_list.assert_called_with(
            opt.output_file_list,
            get_output_file_list_mock.return_value,
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

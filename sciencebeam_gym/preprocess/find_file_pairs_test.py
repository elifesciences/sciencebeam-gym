import logging
import os
from mock import patch

import pytest

import sciencebeam_gym.preprocess.find_file_pairs as find_file_pairs
from sciencebeam_gym.preprocess.find_file_pairs import (
  run,
  parse_args,
  main
)


LOGGER = logging.getLogger(__name__)

BASE_SOURCE_PATH = '/source'

PDF_FILE_1 = BASE_SOURCE_PATH + '/file1.pdf'
XML_FILE_1 = BASE_SOURCE_PATH + '/file1.xml'
PDF_FILE_2 = BASE_SOURCE_PATH + '/file2.pdf'
XML_FILE_2 = BASE_SOURCE_PATH + '/file2.xml'

SOURCE_PATTERN = '*.pdf'
XML_PATTERN = '*.xml'
OUTPUT_FILE = 'file-list.tsv'

SOME_ARGV = [
  '--data-path=%s' % BASE_SOURCE_PATH,
  '--source-pattern=%s' % SOURCE_PATTERN,
  '--xml-pattern=%s' % XML_PATTERN,
  '--out=%s' % OUTPUT_FILE
]


@pytest.fixture(name='find_file_pairs_grouped_by_parent_directory_or_name_mock')
def _find_file_pairs_grouped_by_parent_directory_or_name():
  with patch.object(find_file_pairs, 'find_file_pairs_grouped_by_parent_directory_or_name') as m:
    yield m

@pytest.fixture(name='save_file_pairs_to_csv_mock')
def _save_file_pairs_to_csv():
  with patch.object(find_file_pairs, 'save_file_pairs_to_csv') as m:
    yield m

@pytest.fixture(name='save_file_pairs_to_csv_mock')
def _save_file_pairs_to_csv():
  with patch.object(find_file_pairs, 'save_file_pairs_to_csv') as m:
    yield m

@pytest.fixture(name='parse_args_mock')
def _parse_args():
  with patch.object(find_file_pairs, 'parse_args') as m:
    yield m

@pytest.fixture(name='run_mock')
def _run():
  with patch.object(find_file_pairs, 'run') as m:
    yield m

def _touch(path):
  path.write(b'', ensure=True)
  return path

@pytest.fixture(name='pdf_file_1')
def _pdf_file_1(tmpdir):
  return _touch(tmpdir.join(PDF_FILE_1))

@pytest.fixture(name='xml_file_1')
def _xml_file_1(tmpdir):
  return _touch(tmpdir.join(XML_FILE_1))

@pytest.fixture(name='data_path')
def _data_path(tmpdir):
  return tmpdir.join(BASE_SOURCE_PATH)

@pytest.fixture(name='out_file')
def _out_file(tmpdir):
  return tmpdir.join(OUTPUT_FILE)

class TestRun(object):
  def test_should_pass_around_parameters(
    self,
    find_file_pairs_grouped_by_parent_directory_or_name_mock,
    save_file_pairs_to_csv_mock):

    opt = parse_args(SOME_ARGV)
    find_file_pairs_grouped_by_parent_directory_or_name_mock.return_value = [
      (PDF_FILE_1, XML_FILE_1),
      (PDF_FILE_2, XML_FILE_2)
    ]
    run(opt)
    find_file_pairs_grouped_by_parent_directory_or_name_mock.assert_called_with([
      os.path.join(BASE_SOURCE_PATH, SOURCE_PATTERN),
      os.path.join(BASE_SOURCE_PATH, XML_PATTERN)
    ])
    save_file_pairs_to_csv_mock.assert_called_with(
      opt.out,
      find_file_pairs_grouped_by_parent_directory_or_name_mock.return_value
    )

  def test_should_generate_file_list(self, data_path, pdf_file_1, xml_file_1, out_file):
    LOGGER.debug('pdf_file_1: %s, xml_file: %s', pdf_file_1, xml_file_1)
    opt = parse_args(SOME_ARGV)
    opt.data_path = str(data_path)
    opt.out = str(out_file)
    run(opt)
    out_lines = [s.strip() for s in out_file.read().strip().split('\n')]
    LOGGER.debug('out_lines: %s', out_lines)
    assert out_lines == [
      'source_url\txml_url',
      '%s\t%s' % (pdf_file_1, xml_file_1)
    ]

class TestMain(object):
  def test_should_parse_args_and_call_run(self, parse_args_mock, run_mock):
    main(SOME_ARGV)
    parse_args_mock.assert_called_with(SOME_ARGV)
    run_mock.assert_called_with(parse_args_mock.return_value)

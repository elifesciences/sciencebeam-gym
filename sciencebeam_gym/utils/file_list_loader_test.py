from tempfile import NamedTemporaryFile
from mock import patch

import pytest

import sciencebeam_gym.utils.file_list_loader as file_list_loader
from sciencebeam_gym.utils.file_list_loader import (
  is_csv_or_tsv_file_list,
  load_plain_file_list,
  load_csv_or_tsv_file_list,
  load_file_list
)

FILE_1 = 'file1.pdf'
FILE_2 = 'file2.pdf'
UNICODE_FILE_1 = u'file1\u1234.pdf'

class TestIsCsvOrTsvFileList(object):
  def test_should_return_true_if_file_ext_is_csv(self):
    assert is_csv_or_tsv_file_list('files.csv')

  def test_should_return_true_if_file_ext_is_csv_gz(self):
    assert is_csv_or_tsv_file_list('files.csv.gz')

  def test_should_return_true_if_file_ext_is_tsv(self):
    assert is_csv_or_tsv_file_list('files.tsv')

  def test_should_return_true_if_file_ext_is_tsv_gz(self):
    assert is_csv_or_tsv_file_list('files.tsv.gz')

  def test_should_return_false_if_file_ext_is_lst(self):
    assert not is_csv_or_tsv_file_list('files.lst')

  def test_should_return_false_if_file_ext_is_lst_gz(self):
    assert not is_csv_or_tsv_file_list('files.lst.gz')

class TestLoadPlainFileList(object):
  def test_should_read_multiple_file_paths_from_file(self):
    with NamedTemporaryFile() as f:
      f.write('\n'.join([FILE_1, FILE_2]))
      f.flush()
      assert load_plain_file_list(f.name) == [FILE_1, FILE_2]

  def test_should_read_unicode_file(self):
    with NamedTemporaryFile() as f:
      f.write('\n'.join([UNICODE_FILE_1.encode('utf-8')]))
      f.flush()
      assert load_plain_file_list(f.name) == [UNICODE_FILE_1]

class TestLoadCsvOrTsvFileList(object):
  def test_should_read_multiple_file_paths_from_file_with_header_using_column_name(self):
    with NamedTemporaryFile() as f:
      f.write('\n'.join(['url', FILE_1, FILE_2]))
      f.flush()
      assert load_csv_or_tsv_file_list(f.name, 'url') == [FILE_1, FILE_2]

  def test_should_read_multiple_file_paths_from_file_with_header_using_column_index(self):
    with NamedTemporaryFile() as f:
      f.write('\n'.join(['url', FILE_1, FILE_2]))
      f.flush()
      assert load_csv_or_tsv_file_list(f.name, 0) == [FILE_1, FILE_2]

  def test_should_read_multiple_file_paths_from_file_without_header(self):
    with NamedTemporaryFile() as f:
      f.write('\n'.join([FILE_1, FILE_2]))
      f.flush()
      assert load_csv_or_tsv_file_list(f.name, 0, header=False) == [FILE_1, FILE_2]

  def test_should_read_unicode_file(self):
    with NamedTemporaryFile() as f:
      f.write('\n'.join(['url', UNICODE_FILE_1.encode('utf-8')]))
      f.flush()
      assert load_csv_or_tsv_file_list(f.name, 'url') == [UNICODE_FILE_1]

  def test_should_raise_exception_if_column_name_is_invalid(self):
    with pytest.raises(ValueError):
      with NamedTemporaryFile() as f:
        f.write('\n'.join(['url', FILE_1, FILE_2]))
        f.flush()
        assert load_csv_or_tsv_file_list(f.name, 'xyz') == [FILE_1, FILE_2]

  def test_should_raise_exception_if_column_index_is_invalid(self):
    with pytest.raises(IndexError):
      with NamedTemporaryFile() as f:
        f.write('\n'.join(['url', FILE_1, FILE_2]))
        f.flush()
        assert load_csv_or_tsv_file_list(f.name, 1) == [FILE_1, FILE_2]

class TestLoadFileList(object):
  def test_should_call_load_plain_file_list(self):
    with patch.object(file_list_loader, 'load_plain_file_list') as mock:
      result = load_file_list('file-list.lst', column='url', header=True)
      mock.assert_called_with('file-list.lst')
      assert result == mock.return_value

  def test_should_call_load_csv_or_tsv_file_list(self):
    with patch.object(file_list_loader, 'load_csv_or_tsv_file_list') as mock:
      result = load_file_list('file-list.csv', column='url', header=True)
      mock.assert_called_with('file-list.csv', column='url', header=True)
      assert result == mock.return_value

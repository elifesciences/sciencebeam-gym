import os
from tempfile import NamedTemporaryFile
from mock import patch
from backports.tempfile import TemporaryDirectory

import pytest

import sciencebeam_gym.utils.file_list as file_list_loader
from sciencebeam_gym.utils.file_list import (
  is_csv_or_tsv_file_list,
  load_plain_file_list,
  load_csv_or_tsv_file_list,
  load_file_list,
  save_plain_file_list,
  save_csv_or_tsv_file_list,
  save_file_list
)

FILE_1 = 'file1.pdf'
FILE_2 = 'file2.pdf'
UNICODE_FILE_1 = u'file1\u1234.pdf'
FILE_LIST = [FILE_1, FILE_2]

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

  def test_should_apply_limit(self):
    with NamedTemporaryFile() as f:
      f.write('\n'.join([FILE_1, FILE_2]))
      f.flush()
      assert load_plain_file_list(f.name, limit=1) == [FILE_1]

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

  def test_should_apply_limit(self):
    with NamedTemporaryFile() as f:
      f.write('\n'.join(['url', FILE_1, FILE_2]))
      f.flush()
      assert load_csv_or_tsv_file_list(f.name, 'url', limit=1) == [FILE_1]

class TestLoadFileList(object):
  def test_should_call_load_plain_file_list(self):
    with patch.object(file_list_loader, 'load_plain_file_list') as mock:
      result = load_file_list('file-list.lst', column='url', header=True, limit=1)
      mock.assert_called_with('file-list.lst', limit=1)
      assert result == mock.return_value

  def test_should_call_load_csv_or_tsv_file_list(self):
    with patch.object(file_list_loader, 'load_csv_or_tsv_file_list') as mock:
      result = load_file_list('file-list.csv', column='url', header=True, limit=1)
      mock.assert_called_with('file-list.csv', column='url', header=True, limit=1)
      assert result == mock.return_value

class TestSavePlainFileList(object):
  def test_should_write_multiple_file_paths(self):
    with TemporaryDirectory() as path:
      file_list_path = os.path.join(path, 'out.lst')
      save_plain_file_list(file_list_path, [FILE_1, FILE_2])
      assert load_plain_file_list(file_list_path) == [FILE_1, FILE_2]

  def test_should_write_unicode_file(self):
    with TemporaryDirectory() as path:
      file_list_path = os.path.join(path, 'out.lst')
      save_plain_file_list(file_list_path, [UNICODE_FILE_1])
      assert load_plain_file_list(file_list_path) == [UNICODE_FILE_1]

class TestSaveCsvOrTsvFileList(object):
  def test_should_write_multiple_file_paths(self):
    with TemporaryDirectory() as path:
      file_list_path = os.path.join(path, 'out.csv')
      save_csv_or_tsv_file_list(file_list_path, [FILE_1, FILE_2], column='url')
      assert load_csv_or_tsv_file_list(file_list_path, column='url') == [FILE_1, FILE_2]

  def test_should_write_unicode_file(self):
    with TemporaryDirectory() as path:
      file_list_path = os.path.join(path, 'out.lst')
      save_csv_or_tsv_file_list(file_list_path, [UNICODE_FILE_1], column='url')
      assert load_csv_or_tsv_file_list(file_list_path, column='url') == [UNICODE_FILE_1]

class TestSaveFileList(object):
  def test_should_call_save_plain_file_list(self):
    with patch.object(file_list_loader, 'save_plain_file_list') as mock:
      save_file_list('file-list.lst', FILE_LIST, column='url', header=True)
      mock.assert_called_with('file-list.lst', FILE_LIST)

  def test_should_call_save_csv_or_tsv_file_list(self):
    with patch.object(file_list_loader, 'save_csv_or_tsv_file_list') as mock:
      save_file_list('file-list.csv', FILE_LIST, column='url', header=True)
      mock.assert_called_with('file-list.csv', FILE_LIST, column='url', header=True)

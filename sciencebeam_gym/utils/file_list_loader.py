from __future__ import absolute_import

import codecs
import csv

from apache_beam.io.filesystems import FileSystems

from sciencebeam_gym.utils.csv import (
  csv_delimiter_by_filename
)

def is_csv_or_tsv_file_list(file_list_path):
  return '.csv' in file_list_path or '.tsv' in file_list_path

def load_plain_file_list(file_list_path):
  with FileSystems.open(file_list_path) as f:
    return codecs.getreader('utf-8')(f).readlines(keepends=False)

def load_csv_or_tsv_file_list(file_list_path, column, header=True):
  delimiter = csv_delimiter_by_filename(file_list_path)
  with FileSystems.open(file_list_path) as f:
    reader = csv.reader(f, delimiter=delimiter)
    if not header:
      assert isinstance(column, int)
      column_index = column
    else:
      header_row = next(reader)
      if isinstance(column, int):
        column_index = column
      else:
        column_index = header_row.index(column)
    return [x[column_index].decode('utf-8') for x in reader]

def load_file_list(file_list_path, column, header=True):
  if is_csv_or_tsv_file_list(file_list_path):
    return load_csv_or_tsv_file_list(file_list_path, column=column, header=header)
  else:
    return load_plain_file_list(file_list_path)

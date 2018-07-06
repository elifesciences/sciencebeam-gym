from __future__ import absolute_import

import codecs
import csv
import os
from itertools import islice

from apache_beam.io.filesystems import FileSystems

from sciencebeam_gym.utils.csv import (
  csv_delimiter_by_filename
)

from .file_path import (
  relative_path,
  join_if_relative_path
)


def is_csv_or_tsv_file_list(file_list_path):
  return '.csv' in file_list_path or '.tsv' in file_list_path

def load_plain_file_list(file_list_path, limit=None):
  with FileSystems.open(file_list_path) as f:
    lines = (x.rstrip() for x in codecs.getreader('utf-8')(f))
    if limit:
      lines = islice(lines, 0, limit)
    return list(lines)

def load_csv_or_tsv_file_list(file_list_path, column, header=True, limit=None):
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
        try:
          column_index = header_row.index(column)
        except ValueError:
          raise ValueError(
            'column %s not found, available columns: %s' %
            (column, header_row)
          )
    lines = (x[column_index].decode('utf-8') for x in reader)
    if limit:
      lines = islice(lines, 0, limit)
    return list(lines)

def to_absolute_file_list(base_path, file_list):
  return [join_if_relative_path(base_path, s) for s in file_list]

def to_relative_file_list(base_path, file_list):
  return [relative_path(base_path, s) for s in file_list]

def load_file_list(file_list_path, column, header=True, limit=None, to_absolute=True):
  if is_csv_or_tsv_file_list(file_list_path):
    file_list = load_csv_or_tsv_file_list(
      file_list_path, column=column, header=header, limit=limit
    )
  else:
    file_list = load_plain_file_list(file_list_path, limit=limit)
  if to_absolute:
    file_list = to_absolute_file_list(
      os.path.dirname(file_list_path), file_list
    )
  return file_list

def save_plain_file_list(file_list_path, file_list):
  with FileSystems.create(file_list_path) as f:
    f.write('\n'.join(file_list).encode('utf-8'))

def save_csv_or_tsv_file_list(file_list_path, file_list, column, header=True):
  if header:
    file_list = [column] + file_list
  save_plain_file_list(file_list_path, file_list)

def save_file_list(file_list_path, file_list, column, header=True):
  if is_csv_or_tsv_file_list(file_list_path):
    return save_csv_or_tsv_file_list(
      file_list_path, file_list, column=column, header=header
    )
  else:
    return save_plain_file_list(file_list_path, file_list)

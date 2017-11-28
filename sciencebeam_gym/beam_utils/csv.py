from __future__ import absolute_import

import logging
import csv
from io import BytesIO

from six import string_types

import apache_beam as beam
from apache_beam.io.textio import WriteToText, ReadFromText

from sciencebeam_gym.beam_utils.utils import (
  TransformAndLog
)

from sciencebeam_gym.utils.csv import (
  csv_delimiter_by_filename
)

def get_logger():
  return logging.getLogger(__name__)

def DictToList(fields):
  def wrapper(x):
    get_logger().debug('DictToList: %s -> %s', fields, x)
    return [x.get(field) for field in fields]
  return wrapper

def format_csv_rows(rows, delimiter=','):
  get_logger().debug('format_csv_rows, rows: %s', rows)
  out = BytesIO()
  writer = csv.writer(out, delimiter=delimiter)
  writer.writerows([
    [
      x.encode('utf-8') if isinstance(x, string_types) else x
      for x in row
    ]
    for row in rows
  ])
  result = out.getvalue().decode('utf-8').rstrip('\r\n')
  get_logger().debug('format_csv_rows, result: %s', result)
  return result

class WriteDictCsv(beam.PTransform):
  def __init__(self, path, columns, file_name_suffix=None):
    super(WriteDictCsv, self).__init__()
    self.path = path
    self.columns = columns
    self.file_name_suffix = file_name_suffix
    self.delimiter = csv_delimiter_by_filename(path + file_name_suffix)

  def expand(self, pcoll):
    return (
      pcoll |
      "ToList" >> beam.Map(DictToList(self.columns)) |
      "Format" >> TransformAndLog(
        beam.Map(lambda x: format_csv_rows([x], delimiter=self.delimiter)),
        log_prefix='formatted csv: ',
        log_level='debug'
      ) |
      "Utf8Encode" >> beam.Map(lambda x: x.encode('utf-8')) |
      "Write" >> WriteToText(
        self.path,
        file_name_suffix=self.file_name_suffix,
        header=format_csv_rows([self.columns], delimiter=self.delimiter).encode('utf-8')
      )
    )

def _strip_quotes(s):
  return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s

class ReadDictCsv(beam.PTransform):
  """
  Simplified CSV parser, which does not support:
  * multi-line values
  * delimiter within value
  """
  def __init__(self, filename, header=True, limit=None):
    super(ReadDictCsv, self).__init__()
    if not header:
      raise RuntimeError('header required')
    self.filename = filename
    self.columns = None
    self.delimiter = csv_delimiter_by_filename(filename)
    self.limit = limit
    self.row_num = 0

  def parse_line(self, line):
    if self.limit and self.row_num >= self.limit:
      return
    get_logger().debug('line: %s', line)
    if line:
      row = [
        _strip_quotes(x)
        for x in line.split(self.delimiter)
      ]
      if not self.columns:
        self.columns = row
      else:
        self.row_num += 1
        yield {
          k: x
          for k, x in zip(self.columns, row)
        }

  def expand(self, pcoll):
    return (
      pcoll |
      "Read" >> ReadFromText(self.filename) |
      "Parse" >> beam.FlatMap(self.parse_line)
    )

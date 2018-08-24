import argparse
import csv
import logging
from math import trunc
from random import shuffle

from apache_beam.io.filesystems import FileSystems

from sciencebeam_utils.utils.csv import (
  csv_delimiter_by_filename,
  write_csv_rows
)

from sciencebeam_gym.preprocess.preprocessing_utils import (
  strip_ext,
  get_ext
)

def get_logger():
  return logging.getLogger(__name__)

def extract_proportions_from_args(args):
  digits = 3
  proportions = [
    (name, round(p, digits))
    for name, p in [
      ('train', args.train),
      ('test', args.test),
      ('validation', args.validation)
    ]
    if p > 0
  ]
  if sum(p for _, p in proportions) > 1.0:
    raise ValueError('proportions add up to more than 1.0')
  if not args.test:
    proportions.append(('test', 1.0 - sum(p for _, p in proportions)))
  elif not args.validation:
    proportions.append(('validation', round(1.0 - sum(p for _, p in proportions), digits)))
  proportions = [(name, p) for name, p in proportions if p > 0]
  return proportions

def split_rows(rows, percentages, fill=False):
  size = len(rows)
  chunk_size_list = [int(trunc(p * size)) for p in percentages]
  if fill:
    chunk_size_list[-1] = size - sum(chunk_size_list[:-1])
  chunk_offset_list = [0]
  for chunk_size in chunk_size_list[0:-1]:
    chunk_offset_list.append(chunk_offset_list[-1] + chunk_size)
  get_logger().debug('chunk_offset_list: %s', chunk_offset_list)
  get_logger().debug('chunk_size_list: %s', chunk_size_list)
  return [
    rows[chunk_offset:chunk_offset + chunk_size]
    for chunk_offset, chunk_size in zip(chunk_offset_list, chunk_size_list)
  ]

def output_filenames_for_names(names, prefix, ext):
  return [
    prefix + ('' if prefix.endswith('/') else '-') + name + ext
    for name in names
  ]

def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input', type=str, required=True,
    help='input csv/tsv file'
  )
  parser.add_argument(
    '--train', type=float, required=True,
    help='Train dataset proportion'
  )
  parser.add_argument(
    '--test', type=float, required=False,
    help='Test dataset proportion (if not specified it is assumed to be the remaining percentage)'
  )
  parser.add_argument(
    '--validation', type=float, required=False,
    help='Validation dataset proportion (requires test-proportion)'
  )
  parser.add_argument(
    '--random', action='store_true', default=False,
    help='randomise samples before doing the split'
  )
  parser.add_argument(
    '--fill', action='store_true', default=False,
    help='use up all of the remaining data rows for the last set'
  )
  parser.add_argument(
    '--no-header', action='store_true', default=False,
    help='input file does not contain a header'
  )
  parser.add_argument(
    '--out', type=str, required=False,
    help='output csv/tsv file prefix or directory (if ending with slash)'
    ' will use input file name by default'
  )
  return parser.parse_args(argv)

def process_args(args):
  if not args.out:
    args.out = strip_ext(args.input)

def main(argv=None):
  args = parse_args(argv)
  process_args(args)
  ext = get_ext(args.input)
  proportions = extract_proportions_from_args(args)
  output_filenames = output_filenames_for_names(
    [name for name, _ in proportions],
    args.out,
    ext
  )
  get_logger().info('proportions: %s', proportions)
  get_logger().info('output_filenames: %s', output_filenames)
  delimiter = csv_delimiter_by_filename(args.input)
  with FileSystems.open(args.input) as f:
    reader = csv.reader(f, delimiter=delimiter)
    header_row = None if args.no_header else next(reader)
    data_rows = list(reader)
  get_logger().info('number of rows: %d', len(data_rows))
  if args.random:
    shuffle(data_rows)
  data_rows_by_set = split_rows(
    data_rows,
    [p for _, p in proportions],
    fill=args.fill
  )

  mime_type = 'text/tsv' if delimiter == '\t' else 'text/csv'
  for output_filename, set_data_rows in zip(output_filenames, data_rows_by_set):
    get_logger().info('set size: %d (%s)', len(set_data_rows), output_filename)
    with FileSystems.create(output_filename, mime_type=mime_type) as f:
      writer = csv.writer(f, delimiter=delimiter)
      if header_row:
        write_csv_rows(writer, [header_row])
      write_csv_rows(writer, set_data_rows)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

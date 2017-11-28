import argparse
import csv
import logging

from apache_beam.io.filesystems import FileSystems

from sciencebeam_gym.utils.csv import (
  csv_delimiter_by_filename,
  write_csv_rows
)

from sciencebeam_gym.beam_utils.io import (
  dirname,
  mkdirs_if_not_exists
)

from sciencebeam_gym.preprocess.preprocessing_utils import (
  find_file_pairs_grouped_by_parent_directory_or_name,
  join_if_relative_path
)

def get_logger():
  return logging.getLogger(__name__)

def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data-path', type=str, required=True,
    help='base data path'
  )
  parser.add_argument(
    '--pdf-pattern', type=str, required=True,
    help='pdf pattern'
  )
  parser.add_argument(
    '--xml-pattern', type=str, required=True,
    help='xml pattern'
  )
  parser.add_argument(
    '--out', type=str, required=True,
    help='output csv/tsv file'
  )
  return parser.parse_args(argv)

def main(argv=None):
  args = parse_args(argv)
  get_logger().info('finding file pairs')
  pdf_xml_pairs = find_file_pairs_grouped_by_parent_directory_or_name([
    join_if_relative_path(args.data_path, args.pdf_pattern),
    join_if_relative_path(args.data_path, args.xml_pattern)
  ])
  pdf_xml_pairs = list(pdf_xml_pairs)

  mkdirs_if_not_exists(dirname(args.out))
  delimiter = csv_delimiter_by_filename(args.out)
  mime_type = 'text/tsv' if delimiter == '\t' else 'text/csv'
  with FileSystems.create(args.out, mime_type=mime_type) as f:
    writer = csv.writer(f, delimiter=delimiter)
    write_csv_rows(writer, [['pdf_url', 'xml_url']])
    write_csv_rows(writer, pdf_xml_pairs)
  get_logger().info('written results to %s', args.out)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

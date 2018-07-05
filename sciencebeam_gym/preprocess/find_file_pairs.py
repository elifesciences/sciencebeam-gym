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
    '--source-pattern', type=str, required=True,
    help='source pattern'
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


def save_file_pairs_to_csv(output_path, source_xml_pairs):
  mkdirs_if_not_exists(dirname(output_path))
  delimiter = csv_delimiter_by_filename(output_path)
  mime_type = 'text/tsv' if delimiter == '\t' else 'text/csv'
  with FileSystems.create(output_path, mime_type=mime_type) as f:
    writer = csv.writer(f, delimiter=delimiter)
    write_csv_rows(writer, [['source_url', 'xml_url']])
    write_csv_rows(writer, source_xml_pairs)
  get_logger().info('written results to %s', output_path)

def run(args):
  get_logger().info('finding file pairs')
  source_xml_pairs = find_file_pairs_grouped_by_parent_directory_or_name([
    join_if_relative_path(args.data_path, args.source_pattern),
    join_if_relative_path(args.data_path, args.xml_pattern)
  ])
  source_xml_pairs = list(source_xml_pairs)

  save_file_pairs_to_csv(args.out, source_xml_pairs)

def main(argv=None):
  args = parse_args(argv)
  run(args)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

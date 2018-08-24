import argparse
import csv
import logging

from apache_beam.io.filesystems import FileSystems

from sciencebeam_utils.beam_utils.io import (
  dirname,
  mkdirs_if_not_exists
)

from sciencebeam_gym.utils.csv import (
  csv_delimiter_by_filename,
  write_csv_rows
)

from sciencebeam_gym.utils.file_path import (
  join_if_relative_path,
  relative_path
)

from sciencebeam_gym.preprocess.preprocessing_utils import (
  find_file_pairs_grouped_by_parent_directory_or_name
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

  parser.add_argument(
    '--use-relative-paths', action='store_true',
    help='create a file list with relative paths (relative to the data path)'
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

def to_relative_file_pairs(base_path, file_pairs):
  return (
    (relative_path(base_path, source_url), relative_path(base_path, xml_url))
    for source_url, xml_url in file_pairs
  )

def run(args):
  get_logger().info('finding file pairs')
  source_xml_pairs = find_file_pairs_grouped_by_parent_directory_or_name([
    join_if_relative_path(args.data_path, args.source_pattern),
    join_if_relative_path(args.data_path, args.xml_pattern)
  ])

  if args.use_relative_paths:
    source_xml_pairs = to_relative_file_pairs(args.data_path, source_xml_pairs)

  source_xml_pairs = list(source_xml_pairs)

  save_file_pairs_to_csv(args.out, source_xml_pairs)

def main(argv=None):
  args = parse_args(argv)
  run(args)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

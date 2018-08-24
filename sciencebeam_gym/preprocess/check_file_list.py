from __future__ import division

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

from apache_beam.io.filesystems import FileSystems

from sciencebeam_utils.utils.file_list import (
  load_file_list
)

def get_logger():
  return logging.getLogger(__name__)

def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    'Check file list'
  )

  source = parser.add_argument_group('source')
  source.add_argument(
    '--file-list', type=str, required=True,
    help='path to source file list (tsv/csv/lst)'
  )
  source.add_argument(
    '--file-column', type=str, required=False,
    default='url',
    help='csv/tsv column (ignored for plain file list)'
  )

  parser.add_argument(
    '--limit', type=int, required=False,
    help='limit the files to process'
  )

  parser.add_argument(
    '--debug', action='store_true', default=False,
    help='enable debug output'
  )
  return parser.parse_args(argv)

def map_file_list_to_file_exists(file_list):
  with ThreadPoolExecutor(max_workers=50) as executor:
    return list(executor.map(FileSystems.exists, file_list))

def format_file_exists_results(file_exists):
  if not file_exists:
    return 'empty file list'
  file_exists_count = sum(file_exists)
  file_missing_count = len(file_exists) - file_exists_count
  return (
    'files exist: %d (%.0f%%), files missing: %d (%.0f%%)' %
    (
      file_exists_count, 100.0 * file_exists_count / len(file_exists),
      file_missing_count, 100.0 * file_missing_count / len(file_exists)
    )
  )

def check_files_and_report_result(file_list):
  file_exists = map_file_list_to_file_exists(file_list)
  get_logger().info('%s', format_file_exists_results(file_exists))
  assert sum(file_exists) > 0

def run(opt):
  file_list = load_file_list(
    opt.file_list,
    column=opt.file_column,
    limit=opt.limit
  )
  check_files_and_report_result(file_list)

def main(argv=None):
  args = parse_args(argv)

  if args.debug:
    logging.getLogger().setLevel('DEBUG')

  run(args)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')
  logging.getLogger('oauth2client').setLevel('WARNING')

  main()

import argparse
import logging

from sciencebeam_gym.utils.file_list import (
  load_file_list,
  save_file_list
)

from sciencebeam_gym.preprocess.preprocessing_utils import (
  get_or_validate_base_path,
  get_output_file
)

def get_logger():
  return logging.getLogger(__name__)

def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    'Get output files based on source files and suffix.'
  )

  source = parser.add_argument_group('source')
  source.add_argument(
    '--source-file-list', type=str, required=True,
    help='path to source file list (tsv/csv/lst)'
  )
  source.add_argument(
    '--source-file-column', type=str, required=False,
    default='url',
    help='csv/tsv column (ignored for plain file list)'
  )
  source.add_argument(
    '--source-base-path', type=str, required=False,
    help='base data path for source file urls'
  )

  output = parser.add_argument_group('output')
  output.add_argument(
    '--output-file-list', type=str, required=True,
    help='path to output file list (tsv/csv/lst)'
  )
  output.add_argument(
    '--output-file-column', type=str, required=False,
    default='url',
    help='csv/tsv column (ignored for plain file list)'
  )
  output.add_argument(
    '--output-file-suffix', type=str, required=False,
    help='file suffix (will be added to source urls after removing ext)'
  )
  output.add_argument(
    '--output-base-path', type=str, required=False,
    help='base output path (by default source base path with"-results" suffix)'
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

def get_output_file_list(file_list, source_base_path, output_base_path, output_file_suffix):
  return [
    get_output_file(filename, source_base_path, output_base_path, output_file_suffix)
    for filename in file_list
  ]

def run(opt):
  source_file_list = load_file_list(
    opt.source_file_list,
    column=opt.source_file_column,
    limit=opt.limit
  )
  source_base_path = get_or_validate_base_path(
    source_file_list, opt.source_base_path
  )

  target_file_list = get_output_file_list(
    source_file_list, source_base_path, opt.output_base_path, opt.output_file_suffix
  )

  save_file_list(
    opt.output_file_list,
    target_file_list,
    column=opt.output_file_column
  )

def process_args(args):
  if not args.output_base_path:
    args.output_base_path = args.source_base_path + '-results'

def main(argv=None):
  args = parse_args(argv)
  process_args(args)

  if args.debug:
    logging.getLogger().setLevel('DEBUG')

  run(args)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

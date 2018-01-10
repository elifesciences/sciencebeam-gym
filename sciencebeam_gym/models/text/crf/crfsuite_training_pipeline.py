import logging
import argparse
import pickle
from functools import partial

from six import raise_from

from tqdm import tqdm

from sciencebeam_gym.utils.file_list import (
  load_file_list
)

from sciencebeam_gym.structured_document import (
  merge_token_tag
)

from sciencebeam_gym.structured_document.structured_document_loader import (
  load_structured_document
)

from sciencebeam_gym.preprocess.preprocessing_utils import (
  parse_page_range
)

from sciencebeam_gym.models.text.feature_extractor import (
  structured_document_to_token_props,
  token_props_list_to_features,
  token_props_list_to_labels
)

from sciencebeam_gym.models.text.crf.crfsuite_model import (
  CrfSuiteModel
)

from sciencebeam_gym.beam_utils.io import (
  save_file_content
)

CV_TAG_SCOPE = 'cv'

def get_logger():
  return logging.getLogger(__name__)

def parse_args(argv=None):
  parser = argparse.ArgumentParser('Trains the CRF Suite model')
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

  cv_source = parser.add_argument_group('CV source')
  cv_source.add_argument(
    '--cv-source-file-list', type=str, required=False,
    help='path to cv source file list (tsv/csv/lst)'
    ' (must be in line with main source file list)'
  )
  source.add_argument(
    '--cv-source-file-column', type=str, required=False,
    default='url',
    help='csv/tsv column (ignored for plain file list)'
  )

  parser.add_argument(
    '--limit', type=int, required=False,
    help='limit the files to process'
  )
  parser.add_argument(
    '--pages', type=parse_page_range, default=None,
    help='only processes the selected pages'
  )

  output = parser.add_argument_group('output')
  output.add_argument(
    '--output-path', type=str, required=True,
    help='output path to model'
  )

  parser.add_argument(
    '--debug', action='store_true', default=False,
    help='enable debug output'
  )

  return parser.parse_args(argv)

def load_and_convert_to_token_props(filename, cv_filename, page_range=None):
  try:
    structured_document = load_structured_document(filename, page_range=page_range)
    if cv_filename:
      cv_structured_document = load_structured_document(cv_filename, page_range=page_range)
      structured_document.merge_with(
        cv_structured_document,
        partial(
          merge_token_tag,
          target_scope=CV_TAG_SCOPE
        )
      )
    return list(structured_document_to_token_props(
      structured_document
    ))
  except StandardError as e:
    raise_from(RuntimeError('failed to process %s' % filename), e)

def serialize_model(model):
  return pickle.dumps(model)

def train_model(file_list, cv_file_list, page_range=None, progress=True):
  if not cv_file_list:
    cv_file_list = [None] * len(file_list)

  token_props_list_by_document = []
  total = len(file_list)
  with tqdm(total=total, leave=False, desc='loading files', disable=not progress) as pbar:
    for filename, cv_filename in zip(file_list, cv_file_list):
      token_props_list_by_document.append(
        load_and_convert_to_token_props(filename, cv_filename, page_range=page_range)
      )
      pbar.update(1)
  X = [token_props_list_to_features(x) for x in token_props_list_by_document]
  y = [token_props_list_to_labels(x) for x in token_props_list_by_document]
  model = CrfSuiteModel()
  model.fit(X, y)
  return serialize_model(model)

def save_model(output_filename, model_bytes):
  save_file_content(output_filename, model_bytes)

def run(opt):
  file_list = load_file_list(
    opt.source_file_list,
    opt.source_file_column,
    limit=opt.limit
  )
  if opt.cv_source_file_list:
    cv_file_list = load_file_list(
      opt.cv_source_file_list,
      opt.cv_source_file_column,
      limit=opt.limit
    )
  else:
    cv_file_list = None
  get_logger().info(
    'training using %d files (limit %d), page range: %s',
    len(file_list), opt.limit, opt.pages
  )
  save_model(
    opt.output_path,
    train_model(file_list, cv_file_list, page_range=opt.pages)
  )

def main(argv=None):
  args = parse_args(argv)

  if args.debug:
    logging.getLogger().setLevel('DEBUG')

  run(args)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')
  logging.getLogger('oauth2client').setLevel('WARN')

  main()

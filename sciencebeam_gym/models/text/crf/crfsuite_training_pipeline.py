import logging
import argparse
import pickle
import concurrent
from concurrent.futures import ThreadPoolExecutor

from six import raise_from

from tqdm import tqdm

from sciencebeam_gym.utils.stopwatch import (
  StopWatchRecorder
)

from sciencebeam_gym.utils.file_list import (
  load_file_list
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
  token_props_list_to_labels,
  merge_with_cv_structured_document,
  CV_TAG_SCOPE
)

from sciencebeam_gym.models.text.crf.crfsuite_model import (
  CrfSuiteModel
)

from sciencebeam_gym.beam_utils.io import (
  save_file_content
)

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
  source.add_argument(
    '--cv-source-tag-scope', type=str, required=False,
    default=CV_TAG_SCOPE,
    help='source tag scope to get the cv tag from'
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

def load_and_convert_to_token_props(filename, cv_filename, cv_source_tag_scope, page_range=None):
  try:
    structured_document = load_structured_document(filename, page_range=page_range)
    if cv_filename:
      cv_structured_document = load_structured_document(cv_filename, page_range=page_range)
      structured_document = merge_with_cv_structured_document(
        structured_document,
        cv_structured_document,
        cv_source_tag_scope=cv_source_tag_scope
      )
    return list(structured_document_to_token_props(
      structured_document
    ))
  except StandardError as e:
    raise_from(RuntimeError('failed to process %s (due to %s: %s)' % (filename, type(e), e)), e)

def serialize_model(model):
  return pickle.dumps(model)

def submit_all(executor, fn, iterable):
  return {executor.submit(fn, x) for x in iterable}

def load_token_props_list_by_document(
  file_list, cv_file_list, cv_source_tag_scope, page_range=None, progress=True):

  if not cv_file_list:
    cv_file_list = [None] * len(file_list)

  token_props_list_by_document = []
  total = len(file_list)
  error_count = 0
  with tqdm(total=total, leave=False, desc='loading files', disable=not progress) as pbar:
    with ThreadPoolExecutor(max_workers=50) as executor:
      process_fn = lambda (filename, cv_filename): (
        load_and_convert_to_token_props(
          filename, cv_filename, cv_source_tag_scope=cv_source_tag_scope,
          page_range=page_range
        )
      )
      futures = submit_all(executor, process_fn, zip(file_list, cv_file_list))
      for future in concurrent.futures.as_completed(futures):
        try:
          token_props_list_by_document.append(future.result())
        except StandardError as e:
          get_logger().warning(str(e), exc_info=e)
          error_count += 1
        pbar.update(1)
  if error_count:
    get_logger().info(
      'loading error count: %d (loaded: %d)', error_count, len(token_props_list_by_document)
    )
  return token_props_list_by_document

def train_model(
  file_list, cv_file_list, cv_source_tag_scope, page_range=None, progress=True):

  stop_watch_recorder = StopWatchRecorder()
  model = CrfSuiteModel()

  stop_watch_recorder.start('loading files')
  token_props_list_by_document = load_token_props_list_by_document(
    file_list, cv_file_list, cv_source_tag_scope=cv_source_tag_scope,
    page_range=page_range, progress=progress
  )

  assert token_props_list_by_document

  stop_watch_recorder.start('converting to features')
  X = [token_props_list_to_features(x) for x in token_props_list_by_document]
  y = [token_props_list_to_labels(x) for x in token_props_list_by_document]

  get_logger().info('training model (with %d documents)', len(X))
  stop_watch_recorder.start('train')
  model.fit(X, y)

  stop_watch_recorder.start('serialize')
  serialized_model = serialize_model(model)

  stop_watch_recorder.stop()
  get_logger().info('timings: %s', stop_watch_recorder)

  return serialized_model

def save_model(output_filename, model_bytes):
  get_logger().info('saving model to %s', output_filename)
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
    train_model(
      file_list, cv_file_list, cv_source_tag_scope=opt.cv_source_tag_scope,
      page_range=opt.pages
    )
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

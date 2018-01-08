import logging
import argparse
import pickle

from sciencebeam_gym.utils.file_list_loader import (
  load_file_list
)

from sciencebeam_gym.structured_document.structured_document_loader import (
  load_structured_document
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

def parse_args(argv=None):
  parser = argparse.ArgumentParser('Trains the CRF Suite model')
  parser.add_argument(
    '--source-file-list', type=str, required=True,
    help='path to source file list (tsv/csv/lst)'
  )
  parser.add_argument(
    '--source-file-column', type=str, required=False,
    default='url',
    help='csv/tsv column (ignored for plain file list)'
  )

  parser.add_argument(
    '--output-path', type=str, required=True,
    help='output path to model'
  )

  parser.add_argument(
    '--debug', action='store_true', default=False,
    help='enable debug output'
  )

  return parser.parse_args(argv)

def train_model(file_list):
  token_props_list_by_document = [
    list(structured_document_to_token_props(
      load_structured_document(filename)
    ))
    for filename in file_list
  ]
  X = [token_props_list_to_features(x) for x in token_props_list_by_document]
  y = [token_props_list_to_labels(x) for x in token_props_list_by_document]
  model = CrfSuiteModel()
  model.fit(X, y)
  return pickle.dumps(model)

def save_model(output_filename, model_bytes):
  save_file_content(output_filename, model_bytes)

def run(opt):
  file_list = load_file_list(
    opt.source_file_list,
    opt.source_file_column
  )
  save_model(
    opt.output_path,
    train_model(file_list)
  )

def main(argv=None):
  args = parse_args(argv)

  if args.debug:
    logging.getLogger().setLevel('DEBUG')

  run(args)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

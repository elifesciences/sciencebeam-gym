import argparse
import logging
import pickle
from itertools import repeat

from lxml import etree

from sciencebeam_gym.utils.tf import (
  FileIO
)

from sciencebeam_gym.models.text.feature_extractor import (
  structured_document_to_token_props,
  token_props_list_to_features,
  NONE_TAG
)

from sciencebeam_gym.structured_document.lxml import (
  LxmlStructuredDocument
)

def get_logger():
  return logging.getLogger(__name__)

def _iter_tokens(structured_document):
  for page in structured_document.get_pages():
    for line in structured_document.get_lines_of_page(page):
      for token in structured_document.get_tokens_of_line(line):
        yield token

def annotate_structured_document_using_predictions(
  structured_document, predictions, token_props_list=None):
  """
  Annotates the structured document using the predicted tags.

  Args:
    structured_document: the document that will be tagged
    predictions: list of predicted tags
    token_props_list: optional, used to verify that the correct token is being tagged
  """

  if token_props_list is None:
    token_props_list = repeat(None)
  for token, prediction, token_props in zip(
    _iter_tokens(structured_document),
    predictions, token_props_list
    ):

    if token_props:
      assert structured_document.get_text(token) == token_props['text']

    if prediction and prediction != NONE_TAG:
      structured_document.set_tag(token, prediction)

def predict_and_annotate_structured_document(structured_document, model):
  token_props = list(structured_document_to_token_props(structured_document))
  x = token_props_list_to_features(token_props)
  y_pred = model.predict([x])[0]
  annotate_structured_document_using_predictions(structured_document, y_pred, token_props)

def parse_args(argv=None):
  parser = argparse.ArgumentParser('Annotated LXML using CRF model')
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument(
    '--lxml-path', type=str, required=False,
    help='path to lxml document'
  )

  parser.add_argument(
    '--crf-model', type=str, required=True,
    help='path to saved crf model'
  )

  parser.add_argument(
    '--output-path', type=str, required=True,
    help='output path to annotated document'
  )

  parser.add_argument(
    '--debug', action='store_true', default=False,
    help='enable debug output'
  )

  return parser.parse_args(argv)

def main(argv=None):
  args = parse_args(argv)

  if args.debug:
    logging.getLogger().setLevel('DEBUG')

  with FileIO(args.lxml_path, 'rb') as lxml_f:
    structured_document = LxmlStructuredDocument(
      etree.parse(lxml_f)
    )

  with FileIO(args.crf_model, 'rb') as crf_model_f:
    model = pickle.load(crf_model_f)

  predict_and_annotate_structured_document(
    structured_document,
    model
  )

  get_logger().info('writing result to: %s', args.output_path)
  with FileIO(args.output_path, 'w') as out_f:
    out_f.write(etree.tostring(structured_document.root))

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

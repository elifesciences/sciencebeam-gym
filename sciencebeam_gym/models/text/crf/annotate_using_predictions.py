from itertools import repeat

from sciencebeam_gym.models.text.feature_extractor import (
  structured_document_to_token_props,
  token_props_list_to_features
)

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

    if prediction:
      structured_document.set_tag(token, prediction)

def predict_and_annotate_structured_document(structured_document, model):
  token_props = list(structured_document_to_token_props(structured_document))
  x = token_props_list_to_features(token_props)
  y_pred = model.predict([x])[0]
  annotate_structured_document_using_predictions(structured_document, y_pred, token_props)

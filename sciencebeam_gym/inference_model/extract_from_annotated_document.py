import logging

from sciencebeam_gym.structured_document import (
  B_TAG_PREFIX
)

def get_logger():
  return logging.getLogger(__name__)

class ExtractedItem(object):
  def __init__(self, tag, text, tag_prefix=None):
    self.tag = tag
    self.tag_prefix = tag_prefix
    self.text = text

  def extend(self, other_item):
    return ExtractedItem(
      self.tag,
      self.text + '\n' + other_item.text,
      tag_prefix=self.tag_prefix
    )

def get_lines(structured_document):
  for page in structured_document.get_pages():
    for line in structured_document.get_lines_of_page(page):
      yield line

def extract_from_annotated_tokens(structured_document, tokens, tag_scope=None):
  previous_tokens = []
  previous_tag = None
  previous_tag_prefix = None
  for token in tokens:
    tag_prefix, tag = structured_document.get_tag_prefix_and_value(token, scope=tag_scope)
    if not previous_tokens:
      previous_tokens = [token]
      previous_tag = tag
      previous_tag_prefix = tag_prefix
    elif tag == previous_tag and tag_prefix != B_TAG_PREFIX:
      previous_tokens.append(token)
    else:
      yield ExtractedItem(
        previous_tag,
        ' '.join(structured_document.get_text(t) for t in previous_tokens),
        tag_prefix=previous_tag_prefix
      )
      previous_tokens = [token]
      previous_tag = tag
      previous_tag_prefix = tag_prefix
  if previous_tokens:
    yield ExtractedItem(
      previous_tag,
      ' '.join(structured_document.get_text(t) for t in previous_tokens),
        tag_prefix=previous_tag_prefix
    )

def extract_from_annotated_lines(structured_document, lines, tag_scope=None):
  previous_item = None
  for line in lines:
    tokens = structured_document.get_tokens_of_line(line)
    for item in extract_from_annotated_tokens(structured_document, tokens, tag_scope=tag_scope):
      if previous_item is not None:
        if previous_item.tag == item.tag and item.tag_prefix != B_TAG_PREFIX:
          previous_item = previous_item.extend(item)
        else:
          yield previous_item
          previous_item = item
      else:
        previous_item = item
  if previous_item is not None:
    yield previous_item

def extract_from_annotated_document(structured_document, tag_scope=None):
  return extract_from_annotated_lines(
    structured_document, get_lines(structured_document), tag_scope=tag_scope
  )

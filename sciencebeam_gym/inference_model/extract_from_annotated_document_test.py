import logging

from sciencebeam_gym.structured_document import (
  SimpleToken,
  SimpleLine,
  SimpleStructuredDocument
)

from sciencebeam_gym.inference_model.extract_from_annotated_document import (
  extract_from_annotated_document
)

TEXT_1 = 'some text goes here'
TEXT_2 = 'another line another text'
TEXT_3 = 'more to come'
TAG_1 = 'tag1'
TAG_2 = 'tag2'
TAG_3 = 'tag3'

def get_logger():
  return logging.getLogger(__name__)

def with_tag(x, tag):
  if isinstance(x, SimpleToken):
    x.set_tag(tag)
  elif isinstance(x, list):
    return [with_tag(y, tag) for y in x]
  elif isinstance(x, SimpleLine):
    return SimpleLine(with_tag(x.tokens, tag))
  return x

def to_token(token):
  return SimpleToken(token) if isinstance(token, str) else token

def to_tokens(tokens):
  if isinstance(tokens, str):
    tokens = tokens.split(' ')
  return [to_token(t) for t in tokens]

def to_line(tokens):
  return SimpleLine(to_tokens(tokens))

def annotated_tokens(tokens, tag):
  return with_tag(to_tokens(tokens), tag)

def annotated_line(tokens, tag):
  return with_tag(to_line(tokens), tag)

class TestExtractFromAnnotatedDocument(object):
  def test_should_not_fail_on_empty_document(self):
    structured_document = SimpleStructuredDocument()
    extract_from_annotated_document(structured_document)

  def test_should_extract_single_annotated_line(self):
    lines = [annotated_line(TEXT_1, TAG_1)]
    structured_document = SimpleStructuredDocument(lines=lines)
    result = [
      (x.tag, x.text)
      for x in
      extract_from_annotated_document(structured_document)
    ]
    assert result == [(TAG_1, TEXT_1)]

  def test_should_extract_multiple_annotations_on_single_line(self):
    lines = [to_line(
      annotated_tokens(TEXT_1, TAG_1) +
      to_tokens(TEXT_2) +
      annotated_tokens(TEXT_3, TAG_3)
    )]
    structured_document = SimpleStructuredDocument(lines=lines)
    result = [
      (x.tag, x.text)
      for x in
      extract_from_annotated_document(structured_document)
    ]
    assert result == [
      (TAG_1, TEXT_1),
      (None, TEXT_2),
      (TAG_3, TEXT_3)
    ]

  def test_should_combine_multiple_lines(self):
    lines = [
      annotated_line(TEXT_1, TAG_1),
      annotated_line(TEXT_2, TAG_1)
    ]
    structured_document = SimpleStructuredDocument(lines=lines)
    result = [
      (x.tag, x.text)
      for x in
      extract_from_annotated_document(structured_document)
    ]
    get_logger().debug('result: %s', result)
    assert result == [(TAG_1, '\n'.join([TEXT_1, TEXT_2]))]

  def test_should_combine_multiple_lines_separated_by_other_tag(self):
    lines = [
      annotated_line(TEXT_1, TAG_1),
      annotated_line(TEXT_2, TAG_2),
      annotated_line(TEXT_3, TAG_2),
      annotated_line(TEXT_1, TAG_1),
      annotated_line(TEXT_2, TAG_2),
      annotated_line(TEXT_3, TAG_2)
    ]
    structured_document = SimpleStructuredDocument(lines=lines)
    result = [
      (x.tag, x.text)
      for x in
      extract_from_annotated_document(structured_document)
    ]
    get_logger().debug('result: %s', result)
    assert result == [
      (TAG_1, TEXT_1),
      (TAG_2, '\n'.join([TEXT_2, TEXT_3])),
      (TAG_1, TEXT_1),
      (TAG_2, '\n'.join([TEXT_2, TEXT_3]))
    ]

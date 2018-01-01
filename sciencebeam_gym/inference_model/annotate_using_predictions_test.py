import pytest

import numpy as np

from sciencebeam_gym.structured_document import (
  SimpleStructuredDocument,
  SimpleLine,
  SimpleToken
)

from sciencebeam_gym.utils.bounding_box import (
  BoundingBox
)

from sciencebeam_gym.inference_model.annotate_using_predictions import (
  AnnotatedImage,
  annotate_structured_document_using_predicted_images,
  parse_args
)

TAG_1 = 'tag1'

TOKEN_TEXT_1 = 'a token value'

COLOR_1 = (1, 1, 1)
BG_COLOR = (255, 255, 255)

DEFAULT_HEIGHT = 3
DEFAULT_WIDTH = 3
DEFAULT_BOUNDING_BOX = BoundingBox(0, 0, DEFAULT_WIDTH, DEFAULT_HEIGHT)

def filled_image(color, color_map, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
  return AnnotatedImage(
    np.full((height, width, 3), color),
    color_map
  )

def fill_rect(annoted_image, bounding_box, color):
  for y in range(bounding_box.y, bounding_box.y + bounding_box.height):
    for x in range(bounding_box.x, bounding_box.x + bounding_box.width):
      annoted_image.data[y, x] = color

class TestAnnotatedImage(object):
  def test_should_return_zero_tag_probality_if_color_not_in_output(self):
    annotated_image = filled_image(BG_COLOR, {TAG_1: COLOR_1})
    assert annotated_image.get_tag_probabilities_within(
      DEFAULT_BOUNDING_BOX
    ).get(TAG_1) == 0.0

  def test_should_return_one_tag_probality_if_color_is_only_color_in_output(self):
    annotated_image = filled_image(COLOR_1, {TAG_1: COLOR_1})
    assert annotated_image.get_tag_probabilities_within(
      DEFAULT_BOUNDING_BOX
    ).get(TAG_1) == 1.0

  def test_should_return_zero_tag_probality_if_bounding_box_is_empty(self):
    annotated_image = filled_image(BG_COLOR, {TAG_1: COLOR_1})
    assert annotated_image.get_tag_probabilities_within(
      BoundingBox(0, 0, 0, 0)
    ).get(TAG_1) == 0.0

  def test_should_return_zero_tag_probality_if_bounding_box_is_outside_image(self):
    annotated_image = filled_image(BG_COLOR, {TAG_1: COLOR_1})
    assert annotated_image.get_tag_probabilities_within(
      DEFAULT_BOUNDING_BOX.move_by(DEFAULT_WIDTH, 0)
    ).get(TAG_1) == 0.0

class TestAnnotateStructuredDocumentUsingPredictedImages(object):
  def test_should_not_fail_with_empty_document(self):
    structured_document = SimpleStructuredDocument()
    annotate_structured_document_using_predicted_images(
      structured_document,
      []
    )

  def test_should_not_tag_single_token_not_within_prediction(self):
    token_1 = SimpleToken(TOKEN_TEXT_1)
    structured_document = SimpleStructuredDocument(lines=[SimpleLine([token_1])])
    structured_document.set_bounding_box(
      structured_document.get_pages()[0],
      DEFAULT_BOUNDING_BOX
    )
    structured_document.set_bounding_box(token_1, DEFAULT_BOUNDING_BOX)
    annotate_structured_document_using_predicted_images(
      structured_document,
      [filled_image(BG_COLOR, {TAG_1: COLOR_1})]
    )
    assert structured_document.get_tag(token_1) is None

  def test_should_tag_single_token_within_prediction(self):
    token_1 = SimpleToken(TOKEN_TEXT_1)
    structured_document = SimpleStructuredDocument(lines=[SimpleLine([token_1])])
    structured_document.set_bounding_box(
      structured_document.get_pages()[0],
      DEFAULT_BOUNDING_BOX
    )
    structured_document.set_bounding_box(token_1, DEFAULT_BOUNDING_BOX)
    annotate_structured_document_using_predicted_images(
      structured_document,
      [filled_image(COLOR_1, {TAG_1: COLOR_1})]
    )
    assert structured_document.get_tag(token_1) == TAG_1

  def test_should_tag_single_token_within_full_prediction_at_smaller_scale(self):
    token_1 = SimpleToken(TOKEN_TEXT_1)
    structured_document = SimpleStructuredDocument(lines=[SimpleLine([token_1])])
    structured_document.set_bounding_box(
      structured_document.get_pages()[0],
      BoundingBox(0, 0, DEFAULT_WIDTH * 10, DEFAULT_HEIGHT * 10)
    )
    structured_document.set_bounding_box(
      token_1,
      BoundingBox(0, 0, DEFAULT_WIDTH * 10, DEFAULT_HEIGHT * 10)
    )
    annotate_structured_document_using_predicted_images(
      structured_document,
      [filled_image(COLOR_1, {TAG_1: COLOR_1}, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)]
    )
    assert structured_document.get_tag(token_1) == TAG_1

  def test_should_tag_single_token_within_partial_prediction_at_same_scale(self):
    token_1 = SimpleToken(TOKEN_TEXT_1)
    structured_document = SimpleStructuredDocument(lines=[SimpleLine([token_1])])
    structured_document.set_bounding_box(
      structured_document.get_pages()[0],
      DEFAULT_BOUNDING_BOX
    )
    structured_document.set_bounding_box(
      structured_document.get_pages()[0],
      BoundingBox(0, 0, DEFAULT_WIDTH * 10, DEFAULT_HEIGHT * 10)
    )
    structured_document.set_bounding_box(
      token_1,
      BoundingBox(0, 0, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    )
    annotated_image = filled_image(
      BG_COLOR, {TAG_1: COLOR_1},
      width=DEFAULT_WIDTH * 10,
      height=DEFAULT_HEIGHT * 10
    )
    fill_rect(
      annotated_image,
      BoundingBox(0, 0, DEFAULT_WIDTH, DEFAULT_HEIGHT),
      COLOR_1
    )
    annotate_structured_document_using_predicted_images(
      structured_document,
      [annotated_image]
    )
    assert structured_document.get_tag(token_1) == TAG_1

  def test_should_tag_single_token_within_partial_prediction_at_smaller_scale(self):
    token_1 = SimpleToken(TOKEN_TEXT_1)
    structured_document = SimpleStructuredDocument(lines=[SimpleLine([token_1])])
    structured_document.set_bounding_box(
      structured_document.get_pages()[0],
      BoundingBox(0, 0, DEFAULT_WIDTH * 100, DEFAULT_HEIGHT * 100)
    )
    structured_document.set_bounding_box(
      token_1,
      BoundingBox(0, 0, DEFAULT_WIDTH * 10, DEFAULT_HEIGHT * 10)
    )
    annotated_image = filled_image(
      BG_COLOR, {TAG_1: COLOR_1},
      width=DEFAULT_WIDTH * 10,
      height=DEFAULT_HEIGHT * 10
    )
    fill_rect(
      annotated_image,
      BoundingBox(0, 0, DEFAULT_WIDTH, DEFAULT_HEIGHT),
      COLOR_1
    )
    annotate_structured_document_using_predicted_images(
      structured_document,
      [annotated_image]
    )
    assert structured_document.get_tag(token_1) == TAG_1

class TestParseArgs(object):
  def test_should_raise_error_if_not_enough_arguments_are_passed(self):
    with pytest.raises(SystemExit):
      parse_args([])

  def test_should_not_raise_error_with_minimum_args(self):
    parse_args(['--lxml-path=test', '--images-path=test', '--output-path=test'])

  # def test_should_raise_error_if_mutliple_source_args_are_specified(self):
  #   with pytest.raises(SystemExit):
  #     parse_args([
  #       '--lxml-path=test', '--svg-path=test', '--images-path=test', '--output-path=test'
  #     ])

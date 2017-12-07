from collections import namedtuple
from mock import patch

import pytest
from pytest import raises

import sciencebeam_gym.trainer.models.pix2pix.pix2pix_model as pix2pix_model

from sciencebeam_gym.trainer.models.pix2pix.pix2pix_model import (
  parse_color_map,
  color_map_to_labels,
  color_map_to_colors,
  color_map_to_colors_and_labels,
  colors_and_labels_with_unknown_class,
  UNKNOWN_COLOR,
  UNKNOWN_LABEL,
  Model,
  str_to_list,
  model_args_parser
)

COLOR_MAP_FILENAME = 'color_map.conf'

def some_color(i):
  return (i, i, i)

SOME_COLORS = [some_color(1), some_color(2), some_color(3)]
SOME_LABELS = ['a', 'b', 'c']

class TestParseColorMap(object):
  def test_should_use_fileio_to_load_file_and_pass_to_parser(self):
    with patch.object(pix2pix_model, 'FileIO') as FileIO:
      with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
        parse_color_map(COLOR_MAP_FILENAME)
        FileIO.assert_called_with(COLOR_MAP_FILENAME, 'r')
        parse_color_map_from_file.assert_called_with(FileIO.return_value.__enter__.return_value)

class TestColorMapToLabels(object):
  def test_should_use_color_maps_keys_by_default(self):
    color_map = {
      'a': some_color(1),
      'b': some_color(2),
      'c': some_color(3)
    }
    assert color_map_to_labels(color_map) == ['a', 'b', 'c']

  def test_should_return_specified_labels(self):
    color_map = {
      'a': some_color(1),
      'b': some_color(2),
      'c': some_color(3)
    }
    assert color_map_to_labels(color_map, ['b', 'a']) == ['b', 'a']

  def test_should_raise_error_if_specified_label_not_in_color_map(self):
    color_map = {
      'a': some_color(1),
      'c': some_color(3)
    }
    with raises(ValueError):
      color_map_to_labels(color_map, ['a', 'b'])

class TestColorMapToColors(object):
  def test_should_return_colors_for_labels(self):
    color_map = {
      'a': some_color(1),
      'b': some_color(2),
      'c': some_color(3)
    }
    assert color_map_to_colors(color_map, ['a', 'b']) == [
      some_color(1),
      some_color(2)
    ]

class TestColorMapToColorsAndLabels(object):
  def test_should_return_color_and_specified_labels(self):
    color_map = {
      'a': some_color(1),
      'b': some_color(2),
      'c': some_color(3)
    }
    colors, labels = color_map_to_colors_and_labels(color_map, ['a', 'b'])
    assert colors == [some_color(1), some_color(2)]
    assert labels == ['a', 'b']

  def test_should_return_color_using_sorted_keys_if_no_labels_are_specified(self):
    color_map = {
      'a': some_color(1),
      'b': some_color(2),
      'c': some_color(3)
    }
    colors, labels = color_map_to_colors_and_labels(color_map, None)
    assert colors == [some_color(1), some_color(2), some_color(3)]
    assert labels == ['a', 'b', 'c']

class TestColorsAndLabelsWithUnknownClass(object):
  def test_should_not_add_unknown_class_if_not_enabled(self):
    colors_with_unknown, labels_with_unknown = colors_and_labels_with_unknown_class(
      SOME_COLORS,
      SOME_LABELS,
      use_unknown_class=False
    )
    assert colors_with_unknown == SOME_COLORS
    assert labels_with_unknown == SOME_LABELS

  def test_should_add_unknown_class_if_enabled(self):
    colors_with_unknown, labels_with_unknown = colors_and_labels_with_unknown_class(
      SOME_COLORS,
      SOME_LABELS,
      use_unknown_class=True
    )
    assert colors_with_unknown == SOME_COLORS + [UNKNOWN_COLOR]
    assert labels_with_unknown == SOME_LABELS + [UNKNOWN_LABEL]

  def test_should_add_unknown_class_if_colors_are_empty(self):
    colors_with_unknown, labels_with_unknown = colors_and_labels_with_unknown_class(
      [],
      [],
      use_unknown_class=False
    )
    assert colors_with_unknown == [UNKNOWN_COLOR]
    assert labels_with_unknown == [UNKNOWN_LABEL]

def create_args(**kwargs):
  return namedtuple('args', kwargs.keys())(**kwargs)

@pytest.mark.slow
class TestModel(object):
  def test_parse_separate_channels_with_color_map(self):
    with patch.object(pix2pix_model, 'FileIO'):
      with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
        parse_color_map_from_file.return_value = {
          'a': some_color(1),
          'b': some_color(2),
          'c': some_color(3)
        }
        args = create_args(
          color_map=COLOR_MAP_FILENAME,
          channels=['a', 'b'],
          use_separate_channels=True,
          use_unknown_class=True
        )
        model = Model(args)
        assert model.dimension_colors == [some_color(1), some_color(2)]
        assert model.dimension_labels == ['a', 'b']
        assert model.dimension_colors_with_unknown == [some_color(1), some_color(2), UNKNOWN_COLOR]
        assert model.dimension_labels_with_unknown == ['a', 'b', UNKNOWN_LABEL]

class TestStrToList(object):
  def test_should_parse_empty_string_as_empty_list(self):
    assert str_to_list('') == []

  def test_should_parse_blank_string_as_empty_list(self):
    assert str_to_list(' ') == []

  def test_should_parse_comma_separated_list(self):
    assert str_to_list('a,b,c') == ['a', 'b', 'c']

  def test_should_ignore_white_space_around_values(self):
    assert str_to_list(' a , b , c ') == ['a', 'b', 'c']

class Test(object):
  def test_should_parse_channels(self):
    args = model_args_parser().parse_args(['--channels', 'a,b,c'])
    assert args.channels == ['a', 'b', 'c']

  def test_should_set_channels_to_none_by_default(self):
    args = model_args_parser().parse_args([])
    assert args.channels is None

from collections import namedtuple
from mock import patch

import pytest
from pytest import raises

import tensorflow as tf

from sciencebeam_utils.utils.collection import (
  extend_dict
)

from sciencebeam_gym.trainer.models.pix2pix.pix2pix_core import (
  BaseLoss,
  ALL_BASE_LOSS
)

from sciencebeam_gym.trainer.models.pix2pix.pix2pix_core_test import (
  DEFAULT_ARGS as CORE_DEFAULT_ARGS
)

from sciencebeam_gym.trainer.data.examples_test import (
  EXAMPLE_PROPS_1
)

import sciencebeam_gym.trainer.models.pix2pix.pix2pix_model as pix2pix_model

from sciencebeam_gym.trainer.models.pix2pix.pix2pix_model import (
  parse_color_map,
  color_map_to_labels,
  color_map_to_colors,
  colors_and_labels_with_unknown_class,
  UNKNOWN_COLOR,
  UNKNOWN_LABEL,
  DEFAULT_UNKNOWN_CLASS_WEIGHT,
  Model,
  str_to_list,
  model_args_parser,
  class_weights_to_pos_weight
)

COLOR_MAP_FILENAME = 'color_map.conf'
CLASS_WEIGHTS_FILENAME = 'class-weights.json'
DATA_PATH = 'some/where/*.tfrecord'
BATCH_SIZE = 2

def some_color(i):
  return (i, i, i)

SOME_COLORS = [some_color(1), some_color(2), some_color(3)]
SOME_LABELS = ['a', 'b', 'c']
SOME_COLOR_MAP = {
  k: v for k, v in zip(SOME_LABELS, SOME_COLORS)
}
SOME_CLASS_WEIGHTS = {
  k: float(1 + i) for i, k in enumerate(SOME_LABELS)
}

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

class TestClassWeightsToPosWeight(object):
  def test_should_extract_selected_weights(self):
    assert class_weights_to_pos_weight({
      'a': 0.1,
      'b': 0.2,
      'c': 0.3
    }, ['a', 'b'], False) == [0.1, 0.2]

  def test_should_add_zero_if_unknown_class_is_true(self):
    assert class_weights_to_pos_weight({
      'a': 0.1,
      'b': 0.2,
      'c': 0.3
    }, ['a', 'b'], True, DEFAULT_UNKNOWN_CLASS_WEIGHT) == (
      [0.1, 0.2, DEFAULT_UNKNOWN_CLASS_WEIGHT]
    )

DEFAULT_ARGS = extend_dict(
  CORE_DEFAULT_ARGS,
  dict(
    pages=None,
    color_map=None,
    class_weights=None,
    channels=None,
    filter_annotated=False,
    use_separate_channels=False,
    use_unknown_class=False,
    debug=False
  )
)

def create_args(*args, **kwargs):
  d = extend_dict(*list(args) + [kwargs])
  return namedtuple('args', d.keys())(**d)

class TestModel(object):
  def test_parse_separate_channels_with_color_map_without_class_weights(self):
    with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
      parse_color_map_from_file.return_value = {
        'a': some_color(1),
        'b': some_color(2),
        'c': some_color(3)
      }
      args = create_args(
        DEFAULT_ARGS,
        color_map=COLOR_MAP_FILENAME,
        class_weights=None,
        channels=['a', 'b'],
        use_separate_channels=True,
        use_unknown_class=True
      )
      model = Model(args)
      assert model.dimension_colors == [some_color(1), some_color(2)]
      assert model.dimension_labels == ['a', 'b']
      assert model.dimension_colors_with_unknown == [some_color(1), some_color(2), UNKNOWN_COLOR]
      assert model.dimension_labels_with_unknown == ['a', 'b', UNKNOWN_LABEL]
      assert model.pos_weight is None

  def test_parse_separate_channels_with_color_map_and_class_weights(self):
    with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
      with patch.object(pix2pix_model, 'parse_json_file') as parse_json_file:
        parse_color_map_from_file.return_value = {
          'a': some_color(1),
          'b': some_color(2),
          'c': some_color(3)
        }
        parse_json_file.return_value = {
          'a': 0.1,
          'b': 0.2,
          'c': 0.3
        }
        args = create_args(
          DEFAULT_ARGS,
          base_loss=BaseLoss.WEIGHTED_CROSS_ENTROPY,
          color_map=COLOR_MAP_FILENAME,
          class_weights=CLASS_WEIGHTS_FILENAME,
          channels=['a', 'b'],
          use_separate_channels=True,
          use_unknown_class=True
        )
        model = Model(args)
        assert model.dimension_colors == [some_color(1), some_color(2)]
        assert model.dimension_labels == ['a', 'b']
        assert (
          model.dimension_colors_with_unknown == [some_color(1), some_color(2), UNKNOWN_COLOR]
        )
        assert model.dimension_labels_with_unknown == ['a', 'b', UNKNOWN_LABEL]
        assert model.pos_weight == [0.1, 0.2, DEFAULT_UNKNOWN_CLASS_WEIGHT]

  def test_should_only_include_labels_with_non_zero_class_labels_by_default(self):
    with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
      with patch.object(pix2pix_model, 'parse_json_file') as parse_json_file:
        parse_color_map_from_file.return_value = {
          'a': some_color(1),
          'b': some_color(2),
          'c': some_color(3)
        }
        parse_json_file.return_value = {
          'a': 0.1,
          'b': 0.0,
          'c': 0.3
        }
        args = create_args(
          DEFAULT_ARGS,
          base_loss=BaseLoss.WEIGHTED_CROSS_ENTROPY,
          color_map=COLOR_MAP_FILENAME,
          class_weights=CLASS_WEIGHTS_FILENAME,
          use_separate_channels=True,
          use_unknown_class=True
        )
        model = Model(args)
        assert model.dimension_labels == ['a', 'c']
        assert model.dimension_colors == [some_color(1), some_color(3)]
        assert model.pos_weight == [0.1, 0.3, DEFAULT_UNKNOWN_CLASS_WEIGHT]

  def test_should_use_unknown_class_weight_from_configuration(self):
    with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
      with patch.object(pix2pix_model, 'parse_json_file') as parse_json_file:
        parse_color_map_from_file.return_value = SOME_COLOR_MAP
        parse_json_file.return_value = extend_dict(SOME_CLASS_WEIGHTS, {
          'unknown': 0.99
        })
        args = create_args(
          DEFAULT_ARGS,
          base_loss=BaseLoss.WEIGHTED_CROSS_ENTROPY,
          color_map=COLOR_MAP_FILENAME,
          class_weights=CLASS_WEIGHTS_FILENAME,
          use_separate_channels=True,
          use_unknown_class=True
        )
        model = Model(args)
        assert model.pos_weight[-1] == 0.99

  def test_should_not_load_class_weights_for_cross_entropy(self):
    with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
      with patch.object(pix2pix_model, 'parse_json_file'):
        parse_color_map_from_file.return_value = SOME_COLOR_MAP
        args = create_args(
          DEFAULT_ARGS,
          base_loss=BaseLoss.CROSS_ENTROPY,
          color_map=COLOR_MAP_FILENAME,
          class_weights=CLASS_WEIGHTS_FILENAME,
          use_separate_channels=True,
          use_unknown_class=True
        )
        model = Model(args)
        assert model.pos_weight is None

  def test_should_not_load_class_weights_for_sample_weighted_cross_entropy(self):
    with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
      with patch.object(pix2pix_model, 'parse_json_file'):
        parse_color_map_from_file.return_value = SOME_COLOR_MAP
        args = create_args(
          DEFAULT_ARGS,
          base_loss=BaseLoss.SAMPLE_WEIGHTED_CROSS_ENTROPY,
          color_map=COLOR_MAP_FILENAME,
          class_weights=CLASS_WEIGHTS_FILENAME,
          use_separate_channels=True,
          use_unknown_class=True
        )
        model = Model(args)
        assert model.pos_weight is None

@pytest.mark.slow
@pytest.mark.very_slow
class TestModelBuildGraph(object):
  def test_should_build_train_graph_with_defaults(self):
    with tf.Graph().as_default():
      with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
        with patch.object(pix2pix_model, 'get_matching_files'):
          with patch.object(pix2pix_model, 'read_examples') as read_examples:
            parse_color_map_from_file.return_value = SOME_COLOR_MAP
            read_examples.return_value = EXAMPLE_PROPS_1
            args = create_args(
              DEFAULT_ARGS
            )
            model = Model(args)
            model.build_train_graph(DATA_PATH, BATCH_SIZE)

  def test_should_build_train_graph_with_class_weights(self):
    with tf.Graph().as_default():
      with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
        with patch.object(pix2pix_model, 'parse_json_file') as parse_json_file:
          with patch.object(pix2pix_model, 'get_matching_files'):
            with patch.object(pix2pix_model, 'read_examples') as read_examples:
              parse_color_map_from_file.return_value = SOME_COLOR_MAP
              parse_json_file.return_value = SOME_CLASS_WEIGHTS
              read_examples.return_value = EXAMPLE_PROPS_1
              args = create_args(
                DEFAULT_ARGS,
                base_loss=BaseLoss.WEIGHTED_CROSS_ENTROPY,
                color_map=COLOR_MAP_FILENAME,
                class_weights=CLASS_WEIGHTS_FILENAME,
                channels=['a', 'b'],
                use_separate_channels=True,
                use_unknown_class=True
              )
              model = Model(args)
              tensors = model.build_train_graph(DATA_PATH, BATCH_SIZE)
              assert tensors.pos_weight is not None

  def test_should_build_train_graph_with_sample_class_weights(self):
    with tf.Graph().as_default():
      with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
        with patch.object(pix2pix_model, 'parse_json_file') as parse_json_file:
          with patch.object(pix2pix_model, 'get_matching_files'):
            with patch.object(pix2pix_model, 'read_examples') as read_examples:
              parse_color_map_from_file.return_value = SOME_COLOR_MAP
              parse_json_file.return_value = SOME_CLASS_WEIGHTS
              read_examples.return_value = EXAMPLE_PROPS_1
              args = create_args(
                DEFAULT_ARGS,
                base_loss=BaseLoss.SAMPLE_WEIGHTED_CROSS_ENTROPY,
                color_map=COLOR_MAP_FILENAME,
                channels=SOME_LABELS,
                use_separate_channels=True,
                use_unknown_class=True
              )
              model = Model(args)
              tensors = model.build_train_graph(DATA_PATH, BATCH_SIZE)
              n_output_channels = len(SOME_LABELS) + 1
              assert (
                tensors.separate_channel_annotation_tensor.shape.as_list() ==
                [BATCH_SIZE, model.image_height, model.image_width, n_output_channels]
              )
              assert tensors.pos_weight.shape.as_list() == [BATCH_SIZE, 1, 1, n_output_channels]

  def test_should_build_predict_graph_with_defaults(self):
    with tf.Graph().as_default():
      with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
        with patch.object(pix2pix_model, 'get_matching_files'):
          with patch.object(pix2pix_model, 'read_examples') as read_examples:
            parse_color_map_from_file.return_value = SOME_COLOR_MAP
            read_examples.return_value = EXAMPLE_PROPS_1
            args = create_args(
              DEFAULT_ARGS
            )
            model = Model(args)
            tensors = model.build_predict_graph()
            n_output_channels = 3
            assert (
              tensors.pred.shape.as_list() ==
              [None, model.image_height, model.image_width, n_output_channels]
            )

  def test_should_build_predict_graph_with_sample_class_weights(self):
    with tf.Graph().as_default():
      with patch.object(pix2pix_model, 'parse_color_map_from_file') as parse_color_map_from_file:
        with patch.object(pix2pix_model, 'parse_json_file') as parse_json_file:
          with patch.object(pix2pix_model, 'get_matching_files'):
            with patch.object(pix2pix_model, 'read_examples') as read_examples:
              parse_color_map_from_file.return_value = SOME_COLOR_MAP
              parse_json_file.return_value = SOME_CLASS_WEIGHTS
              read_examples.return_value = EXAMPLE_PROPS_1
              args = create_args(
                DEFAULT_ARGS,
                base_loss=BaseLoss.SAMPLE_WEIGHTED_CROSS_ENTROPY,
                color_map=COLOR_MAP_FILENAME,
                channels=SOME_LABELS,
                use_separate_channels=True,
                use_unknown_class=True
              )
              model = Model(args)
              tensors = model.build_predict_graph()
              n_output_channels = len(SOME_LABELS) + 1
              assert (
                tensors.pred.shape.as_list() ==
                [None, model.image_height, model.image_width, n_output_channels]
              )

class TestStrToList(object):
  def test_should_parse_empty_string_as_empty_list(self):
    assert str_to_list('') == []

  def test_should_parse_blank_string_as_empty_list(self):
    assert str_to_list(' ') == []

  def test_should_parse_comma_separated_list(self):
    assert str_to_list('a,b,c') == ['a', 'b', 'c']

  def test_should_ignore_white_space_around_values(self):
    assert str_to_list(' a , b , c ') == ['a', 'b', 'c']

class TestModelArgsParser(object):
  def test_should_parse_channels(self):
    args = model_args_parser().parse_args(['--channels', 'a,b,c'])
    assert args.channels == ['a', 'b', 'c']

  def test_should_set_channels_to_none_by_default(self):
    args = model_args_parser().parse_args([])
    assert args.channels is None

  def test_should_allow_all_base_loss_options(self):
    for base_loss in ALL_BASE_LOSS:
      args = model_args_parser().parse_args(['--base_loss', base_loss])
      assert args.base_loss == base_loss

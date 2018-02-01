from __future__ import division

import logging
import os
from io import BytesIO

from backports.tempfile import TemporaryDirectory

from sciencebeam_gym.utils.num import (
  assert_close,
  assert_all_close
)

from sciencebeam_gym.utils.tfrecord import (
  dict_to_example,
  write_examples_to_tfrecord
)

from sciencebeam_gym.tools.calculate_class_weights import (
  calculate_sample_frequencies,
  iter_calculate_sample_frequencies,
  calculate_median_class_weight,
  calculate_median_weights_for_frequencies,
  calculate_median_class_weights_for_tfrecord_paths_and_colors,
  calculate_median_class_weights_for_tfrecord_paths_and_color_map,
  calculate_efnet_weights_for_frequencies_by_label,
  tf_calculate_efnet_weights_for_frequency_by_label
)

import tensorflow as tf
import numpy as np
from PIL import Image

def color(i):
  return (i, i, i)

COLOR_0 = color(0)
COLOR_1 = color(1)
COLOR_2 = color(2)
COLOR_3 = color(3)

def setup_module():
  logging.basicConfig(level='DEBUG')

def get_logger():
  return logging.getLogger(__name__)

class TestCalculateSampleFrequencies(object):
  def test_should_return_zero_for_single_not_matching_color(self):
    with tf.Session() as session:
      assert session.run(calculate_sample_frequencies([[
        COLOR_0
      ]], [COLOR_1])) == [0.0]

  def test_should_return_one_for_single_matching_color(self):
    with tf.Session() as session:
      assert session.run(calculate_sample_frequencies([[
        COLOR_1
      ]], [COLOR_1])) == [1.0]

  def test_should_return_total_count_for_multiple_all_matching_color(self):
    with tf.Session() as session:
      assert session.run(calculate_sample_frequencies([[
        COLOR_1, COLOR_1, COLOR_1
      ]], [COLOR_1])) == [3.0]

  def test_should_return_total_count_for_multiple_mixed_color(self):
    with tf.Session() as session:
      assert session.run(calculate_sample_frequencies([[
        COLOR_1, COLOR_1, COLOR_2
      ]], [COLOR_1, COLOR_2])) == [2.0, 1.0]

  def test_should_include_unknown_class_count_if_enabled(self):
    with tf.Session() as session:
      assert session.run(calculate_sample_frequencies([[
        COLOR_1, COLOR_2, COLOR_3
      ]], [COLOR_1], use_unknown_class=True)) == [1.0, 2.0]

def encode_png(data):
  out = BytesIO()
  data = np.array(data, dtype=np.uint8)
  image_size = data.shape[:-1]
  get_logger().debug('data type: %s', data.dtype)
  get_logger().debug('image_size: %s', image_size)
  mode = 'RGB'
  image = Image.fromarray(data, mode)
  image.save(out, 'png')
  image_bytes = out.getvalue()
  return image_bytes

class TestIterCalculateSampleFrequencies(object):
  def test_should_return_zero_for_single_not_matching_color(self):
    assert list(iter_calculate_sample_frequencies([
      [[
        COLOR_0
      ]]
    ], [COLOR_1], image_shape=(1, 1, 3))) == [[0.0]]

  def test_should_infer_image_shape(self):
    assert list(iter_calculate_sample_frequencies([
      [[
        COLOR_0
      ]]
    ], [COLOR_1])) == [[0.0]]

  def test_should_include_unknown_class_if_enabled(self):
    assert list(iter_calculate_sample_frequencies([
      [[
        COLOR_0
      ]]
    ], [COLOR_1], image_shape=(1, 1, 3), use_unknown_class=True)) == [[0.0, 1.0]]

  def test_should_include_unknown_class_if_enabled_and_infer_shape(self):
    assert list(iter_calculate_sample_frequencies([
      [[
        COLOR_0
      ]]
    ], [COLOR_1], use_unknown_class=True)) == [[0.0, 1.0]]

  def test_should_return_total_count_for_multiple_mixed_color(self):
    assert list(iter_calculate_sample_frequencies([
      [[
        COLOR_0, COLOR_0, COLOR_0
      ]], [[
        COLOR_0, COLOR_1, COLOR_2
      ]], [[
        COLOR_1, COLOR_1, COLOR_2
      ]]
    ], [COLOR_1, COLOR_2])) == [
      [0.0, 0.0],
      [1.0, 1.0],
      [2.0, 1.0]
    ]

  def test_should_decode_png(self):
    assert list(iter_calculate_sample_frequencies([
      encode_png([[
        COLOR_1
      ]])
    ], [COLOR_1], image_shape=(1, 1, 3), image_format='png')) == [[1.0]]

  def test_should_infer_shape_when_decoding_png(self):
    assert list(iter_calculate_sample_frequencies([
      encode_png([[
        COLOR_1
      ]])
    ], [COLOR_1], image_format='png')) == [[1.0]]

  def test_should_infer_shape_when_decoding_png_and_include_unknown_class(self):
    assert list(iter_calculate_sample_frequencies([
      encode_png([[
        COLOR_1, COLOR_2, COLOR_3
      ]])
    ], [COLOR_1], image_format='png', use_unknown_class=True)) == [[1.0, 2.0]]

class TestTfCalculateEfnetForFrequencyByLabel(object):
  def test_should_return_same_value_for_classes_with_same_frequencies(self):
    with tf.Graph().as_default():
      with tf.Session():
        frequencies = [1, 1]
        result = tf_calculate_efnet_weights_for_frequency_by_label(frequencies).eval()
        assert result[0] == result[1]

  def test_should_return_higher_value_for_less_frequent_occuring_class(self):
    with tf.Graph().as_default():
      with tf.Session():
        frequencies = [2, 1]
        result = tf_calculate_efnet_weights_for_frequency_by_label(frequencies).eval()
        assert result[0] < result[1]

  def test_should_return_zero_value_for_not_occuring_class(self):
    with tf.Graph().as_default():
      with tf.Session():
        frequencies = [1, 0]
        result = tf_calculate_efnet_weights_for_frequency_by_label(frequencies).eval()
        assert result[-1] == 0.0

class TestCalculateEfnetForFrequenciesByLabel(object):
  def test_should_return_same_value_for_classes_with_same_frequencies(self):
    frequencies = [
      [0, 1],
      [0, 1]
    ]
    result = calculate_efnet_weights_for_frequencies_by_label(frequencies)
    assert result[0] == result[1]

  def test_should_return_higher_value_for_less_frequent_occuring_class(self):
    frequencies = [
      [1, 1],
      [0, 1]
    ]
    result = calculate_efnet_weights_for_frequencies_by_label(frequencies)
    assert result[0] < result[1]

  def test_should_return_zero_value_for_not_occuring_class(self):
    frequencies = [
      [1, 1],
      [0, 0]
    ]
    result = calculate_efnet_weights_for_frequencies_by_label(frequencies)
    assert result[-1] == 0.0

class TestCalculateMedianClassWeight(object):
  def test_should_return_median_frequency_balanced_for_same_frequencies(self):
    assert calculate_median_class_weight([3, 3, 3]) == 1 / 3

  def test_should_return_median_frequence_balanced_for_different_frequencies(self):
    assert calculate_median_class_weight([1, 3, 5]) == 1 / 3

  def test_should_return_zero_for_all_zero_frequencies(self):
    assert calculate_median_class_weight([0, 0, 0]) == 0.0

class TestCalculateWeightsForFrequencies(object):
  def test_should_return_one_for_single_class(self):
    assert calculate_median_weights_for_frequencies([
      [3, 3, 3]
    ]) == [1.0]

  def test_should_return_50p_for_classes_with_same_frequencies(self):
    assert calculate_median_weights_for_frequencies([
      [3, 3, 3],
      [3, 3, 3]
    ]) == [0.5, 0.5]

  def test_should_return_higher_value_for_less_frequent_occuring_class(self):
    frequencies = [
      [1, 1],
      [1, 1],
      [0, 1]
    ]
    result = calculate_median_weights_for_frequencies(frequencies)
    get_logger().debug('result: %s', result)
    assert_close(sum(result), 1.0)
    assert_all_close(result, [0.25, 0.25, 0.5], atol=0.001)

  def test_should_return_zero_value_for_not_occuring_class(self):
    frequencies = [
      [1, 1],
      [1, 1],
      [0, 0]
    ]
    result = calculate_median_weights_for_frequencies(frequencies)
    get_logger().debug('result: %s', result)
    assert_close(sum(result), 1.0)
    assert_all_close(result, [0.5, 0.5, 0.0], atol=0.001)

class TestCalculateMedianClassWeightsForFfrecordPathsAndColors(object):
  def test_should_calculate_median_class_weights_for_single_image_and_single_color(self):
    with TemporaryDirectory() as path:
      tfrecord_filename = os.path.join(path, 'data.tfrecord')
      get_logger().debug('writing to test tfrecord_filename: %s', tfrecord_filename)
      write_examples_to_tfrecord(tfrecord_filename, [dict_to_example({
        'image': encode_png([[
          COLOR_1
        ]])
      })])
      class_weights = calculate_median_class_weights_for_tfrecord_paths_and_colors(
        [tfrecord_filename], 'image', [COLOR_1]
      )
      assert class_weights == [1.0]

  def test_should_calculate_median_class_weights_for_multiple_image_and_multiple_images(self):
    with TemporaryDirectory() as path:
      tfrecord_filename = os.path.join(path, 'data.tfrecord')
      get_logger().debug('writing to test tfrecord_filename: %s', tfrecord_filename)
      write_examples_to_tfrecord(tfrecord_filename, [dict_to_example({
        'image': encode_png([[
          COLOR_0, COLOR_1, COLOR_2
        ]])
      }), dict_to_example({
        'image': encode_png([[
          COLOR_1, COLOR_2, COLOR_3
        ]])
      })])
      class_weights = calculate_median_class_weights_for_tfrecord_paths_and_colors(
        [tfrecord_filename], 'image', [COLOR_1, COLOR_2, COLOR_3]
      )
      assert class_weights == [0.25, 0.25, 0.5]

  def test_should_return_zero_for_non_occuring_class(self):
    with TemporaryDirectory() as path:
      tfrecord_filename = os.path.join(path, 'data.tfrecord')
      get_logger().debug('writing to test tfrecord_filename: %s', tfrecord_filename)
      write_examples_to_tfrecord(tfrecord_filename, [dict_to_example({
        'image': encode_png([[
          COLOR_1
        ]])
      })])
      class_weights = calculate_median_class_weights_for_tfrecord_paths_and_colors(
        [tfrecord_filename], 'image', [COLOR_1, COLOR_2]
      )
      assert class_weights == [1.0, 0.0]

class TestCalculateMedianClassWeightsForFfrecordPathsAndColorMap(object):
  def test_should_calculate_median_class_weights_for_single_image_and_single_color(self):
    with TemporaryDirectory() as path:
      tfrecord_filename = os.path.join(path, 'data.tfrecord')
      get_logger().debug('writing to test tfrecord_filename: %s', tfrecord_filename)
      write_examples_to_tfrecord(tfrecord_filename, [dict_to_example({
        'image': encode_png([[
          COLOR_1, COLOR_2
        ]])
      })])
      class_weights_map = calculate_median_class_weights_for_tfrecord_paths_and_color_map(
        [tfrecord_filename], 'image', {
          'color1': COLOR_1,
          'color2': COLOR_2,
          'color3': COLOR_3
        },
        channels=['color1', 'color2']
      )
      assert class_weights_map == {
        'color1': 0.5,
        'color2': 0.5
      }

  def test_should_use_color_map_keys_as_channels_by_default(self):
    with TemporaryDirectory() as path:
      tfrecord_filename = os.path.join(path, 'data.tfrecord')
      get_logger().debug('writing to test tfrecord_filename: %s', tfrecord_filename)
      write_examples_to_tfrecord(tfrecord_filename, [dict_to_example({
        'image': encode_png([[
          COLOR_1, COLOR_2
        ]])
      })])
      class_weights_map = calculate_median_class_weights_for_tfrecord_paths_and_color_map(
        [tfrecord_filename], 'image', {
          'color1': COLOR_1,
          'color2': COLOR_2
        }
      )
      assert set(class_weights_map.keys()) == {'color1', 'color2'}

  def test_should_include_unknown_class_if_enabled(self):
    with TemporaryDirectory() as path:
      tfrecord_filename = os.path.join(path, 'data.tfrecord')
      get_logger().debug('writing to test tfrecord_filename: %s', tfrecord_filename)
      write_examples_to_tfrecord(tfrecord_filename, [dict_to_example({
        'image': encode_png([[
          COLOR_0, COLOR_1, COLOR_2, COLOR_3
        ]])
      })])
      class_weights_map = calculate_median_class_weights_for_tfrecord_paths_and_color_map(
        [tfrecord_filename], 'image', {
          'color1': COLOR_1,
          'color2': COLOR_2
        },
        use_unknown_class=True,
        unknown_class_label='unknown'
      )
      assert set(class_weights_map.keys()) == {'color1', 'color2', 'unknown'}

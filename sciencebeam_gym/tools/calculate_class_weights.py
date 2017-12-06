from __future__ import division
from __future__ import print_function

import argparse
import logging
import json

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io # pylint: disable=E0611
from tqdm import tqdm

from sciencebeam_gym.preprocess.color_map import (
  parse_color_map_from_file
)

from sciencebeam_gym.utils.tfrecord import (
  iter_read_tfrecord_file_as_dict_list
)

def get_logger():
  return logging.getLogger(__name__)

def color_frequency(image, color):
  return tf.reduce_sum(
    tf.cast(
      tf.reduce_all(
        tf.equal(image, color),
        axis=-1,
        name='is_color'
      ),
      tf.float32
    )
  )

def get_shape(x):
  try:
    return x.shape
  except AttributeError:
    return tf.constant(x).shape

def calculate_sample_frequencies(image, colors):
  return [
    color_frequency(image, color)
    for color in colors
  ]

def iter_calculate_sample_frequencies(images, colors, image_shape=None, image_format=None):
  with tf.Graph().as_default():
    if image_format == 'png':
      image_tensor = tf.placeholder(tf.string, shape=[], name='image')
      decoded_image_tensor = tf.image.decode_png(image_tensor, channels=3)
    else:
      if image_shape is None:
        image_shape = (None, None, 3)
      image_tensor = tf.placeholder(tf.uint8, shape=image_shape, name='image')
      decoded_image_tensor = image_tensor
    get_logger().debug('decoded_image_tensor: %s', decoded_image_tensor)
    frequency_tensors = calculate_sample_frequencies(decoded_image_tensor, colors)
    with tf.Session() as session:
      for image in images:
        frequencies = session.run(frequency_tensors, {
          image_tensor: image
        })
        get_logger().debug('frequencies: %s', frequencies)
        yield frequencies

def calculate_median_class_weight(class_frequencies):
  """
  Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c
    where median_freq_c is the median frequency of the class
    for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels
    of the images where c appeared.
  """

  non_zero_frequencies = [f for f in class_frequencies if f != 0.0]
  if not non_zero_frequencies:
    return 0.0
  get_logger().debug('non_zero_frequencies: %s', non_zero_frequencies)
  total_freq_c = sum(non_zero_frequencies)
  get_logger().debug('total_freq_c: %s', total_freq_c)
  median_freq_c = np.median(non_zero_frequencies)
  get_logger().debug('median_freq_c: %s', median_freq_c)
  return median_freq_c / total_freq_c

def calculate_median_weights_for_frequencies(frequencies):
  median_frequencies_balanced = [
    calculate_median_class_weight(f)
    for f in frequencies
  ]
  total = sum(median_frequencies_balanced)
  return [
    f / total
    for f in median_frequencies_balanced
  ]

def parse_color_map(color_map_filename):
  with file_io.FileIO(color_map_filename, 'r') as config_f:
    return parse_color_map_from_file(
      config_f
    )

def transpose(m):
  return zip(*m)

def iter_images_for_tfrecord_paths(tfrecord_paths, image_key, progress=False):
  for tfrecord_path in tfrecord_paths:
    get_logger().info('tfrecord_path: %s', tfrecord_path)
    filenames = file_io.get_matching_files(tfrecord_path)
    with tqdm(list(filenames), leave=False, disable=not progress) as pbar:
      for tfrecord_filename in pbar:
        pbar.set_description('%-40s' % tfrecord_filename)
        get_logger().debug('tfrecord_filename: %s', tfrecord_filename)
        for d in iter_read_tfrecord_file_as_dict_list(tfrecord_filename, keys={image_key}):
          yield d[image_key]

def calculate_median_class_weights_for_tfrecord_paths_and_colors(
  tfrecord_paths, image_key, colors, progress=False):

  get_logger().debug('colors: %s', colors)
  get_logger().info('loading tfrecords: %s', tfrecord_paths)
  images = iter_images_for_tfrecord_paths(tfrecord_paths, image_key, progress=progress)
  if progress:
    images = list(images)
    images = tqdm(images, 'analysing images', leave=False)
  frequency_list = list(iter_calculate_sample_frequencies(images, colors, image_format='png'))
  get_logger().debug('frequency_list: %s', frequency_list)
  frequencies = transpose(frequency_list)
  get_logger().debug('frequencies: %s', frequencies)
  class_weights = calculate_median_weights_for_frequencies(frequencies)
  return class_weights

def calculate_median_class_weights_for_tfrecord_paths_and_color_map(
  tfrecord_paths, image_key, color_map, channels=None, progress=False):
  if not channels:
    channels = sorted(color_map.keys())
  colors = [color_map[k] for k in channels]
  class_weights = calculate_median_class_weights_for_tfrecord_paths_and_colors(
    tfrecord_paths,
    image_key,
    colors,
    progress=progress
  )
  return {
    k: class_weight for k, class_weight in zip(channels, class_weights)
  }

def str_to_list(s):
  s = s.strip()
  if not s:
    return []
  return [x.strip() for x in s.split(',')]

def get_args_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--tfrecord-paths',
    required=True,
    type=str,
    action='append',
    help='The paths to the tf-records files to analyse.'
  )
  parser.add_argument(
    '--image-key',
    required=False,
    type=str,
    help='The name of the image key to do the class weights on.'
  )
  parser.add_argument(
    '--color-map',
    required=True,
    type=str,
    help='The color-map filename.'
  )
  parser.add_argument(
    '--channels',
    type=str_to_list,
    help='The channels to use (subset of color map), otherwise all of the labels will be used'
  )
  parser.add_argument(
    '--out',
    required=False,
    type=str,
    help='The filename the output file (json), otherwise the output will be written to stdout.'
  )
  return parser

def parse_args(argv=None):
  parser = get_args_parser()
  parsed_args = parser.parse_args(argv)
  return parsed_args

def main(argv=None):
  args = parse_args(argv)
  color_map = parse_color_map(args.color_map)
  class_weights_map = calculate_median_class_weights_for_tfrecord_paths_and_color_map(
    args.tfrecord_paths,
    args.image_key,
    color_map,
    channels=args.channels,
    progress=True
  )
  get_logger().info('class_weights: %s', class_weights_map)
  json_str = json.dumps(class_weights_map, indent=2)
  if args.out:
    with file_io.FileIO(args.out, 'wb') as out_f:
      out_f.write(json_str)
  else:
    print(json_str)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')
  main()

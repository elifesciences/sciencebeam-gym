# partially copied from tensorflow example project
from __future__ import absolute_import
import argparse
import datetime
import errno
import io
import logging
import os
import subprocess
import sys
import re

import six

import apache_beam as beam
from apache_beam.metrics import Metrics

# pylint: disable=g-import-not-at-top
# TODO(yxshi): Remove after Dataflow 0.4.5 SDK is released.
try:
  try:
    from apache_beam.options.pipeline_options import PipelineOptions
  except ImportError:
    from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
  from apache_beam.utils.options import PipelineOptions
import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

from PIL import Image

# TODO copied functions due to pickling issue
# from .colorize_image import (
#   parse_color_map,
#   map_colors
# )
import re
from six.moves.configparser import ConfigParser

slim = tf.contrib.slim

error_count = Metrics.counter('main', 'errorCount')

def parse_color_map(f, section_names=None):
  if section_names is None:
    section_names = ['color_map']
  color_map_config = ConfigParser()
  color_map_config.readfp(f)

  num_pattern = re.compile(r'(\d+)')
  rgb_pattern = re.compile(r'\((\d+),(\d+),(\d+)\)')

  def parse_color(s):
    m = num_pattern.match(s)
    if m:
      x = int(m.group(1))
      return (x, x, x)
    else:
      m = rgb_pattern.match(s)
      if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    raise Exception('invalid color value: {}'.format(s))

  color_map = dict()
  for section_name in section_names:
    if color_map_config.has_section(section_name):
      for k, v in color_map_config.items(section_name):
        color_map[parse_color(k)] = parse_color(v)
  return color_map

def map_colors(img, color_map, default_color=None):
  if color_map is None or len(color_map) == 0:
    return img
  original_data = img.getdata()
  mapped_data = [
    color_map.get(color, default_color or color)
    for color in original_data
  ]
  img.putdata(mapped_data)
  return img

def _open_file_read_binary(uri):
  # TF will enable 'rb' in future versions, but until then, 'r' is
  # required.
  try:
    return file_io.FileIO(uri, mode='rb')
  except errors.InvalidArgumentError:
    return file_io.FileIO(uri, mode='r')

def iter_read_image(uri):
  try:
    with _open_file_read_binary(uri) as f:
      image_bytes = f.read()
      yield Image.open(io.BytesIO(image_bytes)).convert('RGB')

  # A variety of different calling libraries throw different exceptions here.
  # They all correspond to an unreadable file so we treat them equivalently.
  except Exception as e:  # pylint: disable=broad-except
    logging.exception('Error processing image %s: %s', uri, str(e))
    error_count.inc()

def image_resize_nearest(image, size):
  return image.resize(size, Image.NEAREST)

def image_resize_bicubic(image, size):
  return image.resize(size, Image.BICUBIC)

def image_save_to_bytes(image, format):
  output = io.BytesIO()
  image.save(output, format)
  return output.getvalue()

def get_image_filenames_for_filenames(filenames, target_pattern):
  r_target_pattern = re.compile(target_pattern)
  png_filenames = [s for s in filenames if s.endswith('.png')]
  annot_filenames = sorted([s for s in png_filenames if r_target_pattern.search(s)])
  image_filenames = sorted([s for s in png_filenames if s not in annot_filenames])
  assert len(annot_filenames) == len(image_filenames)
  return image_filenames, annot_filenames

def get_image_filenames_for_patterns(data_paths, target_pattern):
  files = []
  for path in data_paths:
    files.extend(file_io.get_matching_files(path))
  return get_image_filenames_for_filenames(files, target_pattern)

def MapDictPropAs(in_key, out_key, fn, label='MapDictPropAs'):
  def wrapper_fn(d):
    d = d.copy()
    d.update({
      out_key: fn(d[in_key])
    })
    return d
  return label >> beam.Map(wrapper_fn)

class MapDictPropAsIfNotNone(beam.DoFn):
  def __init__(self, in_key, out_key, fn):
    self.in_key = in_key
    self.out_key = out_key
    self.fn = fn

  def process(self, element):
    if hasattr(element, 'element'):
      element = element.element
    if not isinstance(element, dict):
      raise Exception('expected dict, got: {} ({})'.format(type(element), element))
    d = element
    value = self.fn(d[self.in_key])
    if value is not None:
      d = d.copy()
      d.update({
        self.out_key: value
      })
      yield d

def FlatMapDictProp(in_key, out_key, fn, label='FlatMapDictProp'):
  def wrapper_fn(d):
    out_value = fn(d[in_key])
    for v in out_value:
      d_out = d.copy()
      d_out.update({
        out_key: v
      })
      yield d_out
  return label >> beam.FlatMap(wrapper_fn)

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def WriteToLog(message_fn):
  def wrapper_fn(x):
    logging.info(message_fn(x))
    return x
  return beam.Map(wrapper_fn)

def ReadAndConvertInputImage(image_size):
  def convert_annotation_image(image):
    image = image_resize_bicubic(image, image_size)
    return image_save_to_bytes(image, 'png')
  return lambda uri: [
    convert_annotation_image(image)
    for image in iter_read_image(uri)
  ]

def ReadAndConvertAnnotationImage(image_size, color_map):
  def convert_annotation_image(image):
    image = image_resize_nearest(image, image_size)
    if color_map:
      image = map_colors(image, color_map, default_color=(255, 255, 255))
    return image_save_to_bytes(image, 'png')
  return lambda uri: [
    convert_annotation_image(image)
    for image in iter_read_image(uri)
  ]

def configure_pipeline(p, opt):
  """Specify PCollection and transformations in pipeline."""
  logger = logging.getLogger(__name__)
  image_size = (opt.image_width, opt.image_height)
  color_map = None
  if opt.color_map:
    with file_io.FileIO(opt.color_map, 'r') as config_f:
      color_map = parse_color_map(config_f, section_names=['color_alias', 'color_map'])
  if color_map:
    logger.info('read {} color mappings'.format(len(color_map)))
  else:
    logger.info('no color mappings configured')
  train_image_filenames, train_annotation_filenames = (
    get_image_filenames_for_patterns(opt.data_paths, opt.target_pattern)
  )
  logging.info('train/annotation_filenames:\n%s', '\n'.join([
    '...{} - ...{}'.format(train_image_filename[-20:], train_annotation_filename[-40:])
    for train_image_filename, train_annotation_filename
    in zip(train_image_filenames, train_annotation_filenames)
  ]))

  file_io.recursive_create_dir(opt.output_path)

  _ = (
    p
    | beam.Create([
      {
        'input_uri': train_image_filename,
        'annotation_uri': train_annotation_filename
      }
      for train_image_filename, train_annotation_filename in
      zip(train_image_filenames, train_annotation_filenames)
    ])
    | 'ReadAndConvertInputImage' >> FlatMapDictProp(
      'input_uri', 'input_image', ReadAndConvertInputImage(
        image_size
      )
    )
    | 'ReadAndConvertAnnotationImage' >> FlatMapDictProp(
      'annotation_uri', 'annotation_image', ReadAndConvertAnnotationImage(
        image_size, color_map
      )
    )
    | 'Log' >> WriteToLog(lambda x: 'processed: {} ({})'.format(
      x['input_uri'], x['annotation_uri']
    ))
    | 'ConvertToExamples' >> beam.Map(
      lambda x: tf.train.Example(features=tf.train.Features(feature={
        k: _bytes_feature([v])
        for k, v in six.iteritems(x)
      }))
    )
    | 'SerializeToString' >> beam.Map(lambda x: x.SerializeToString())
    | 'SaveToDisk' >> beam.io.WriteToTFRecord(
      opt.output_path,
      file_name_suffix='.tfrecord.gz'
    )
  )

def run(in_args=None):
  """Runs the pre-processing pipeline."""

  pipeline_options = PipelineOptions.from_dictionary(vars(in_args))
  with beam.Pipeline(options=pipeline_options) as p:
    configure_pipeline(p, in_args)

def get_cloud_project():
  cmd = [
    'gcloud', '-q', 'config', 'list', 'project',
    '--format=value(core.project)'
  ]
  with open(os.devnull, 'w') as dev_null:
    try:
      res = subprocess.check_output(cmd, stderr=dev_null).strip()
      if not res:
        raise Exception(
          '--cloud specified but no Google Cloud Platform '
          'project found.\n'
          'Please specify your project name with the --project '
          'flag or set a default project: '
          'gcloud config set project YOUR_PROJECT_NAME'
        )
      return res
    except OSError as e:
      if e.errno == errno.ENOENT:
        raise Exception(
          'gcloud is not installed. The Google Cloud SDK is '
          'necessary to communicate with the Cloud ML service. '
          'Please install and set up gcloud.'
        )
      raise

def default_args(argv):
  """Provides default values for Workflow flags."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--data_paths',
    type=str,
    action='append',
    help='The paths to the training data files. '
      'Can be comma separated list of files or glob pattern.'
  )
  parser.add_argument(
    '--target_pattern',
    type=str,
    default=r'\btarget\b|\bannot',
    help='The regex pattern to identify target files.'
  )
  parser.add_argument(
    '--output_path',
    required=True,
    help='Output directory to write results to.'
  )
  parser.add_argument(
    '--project',
    type=str,
    help='The cloud project name to be used for running this pipeline'
  )

  parser.add_argument(
    '--job_name',
    type=str,
    default='sciencebeam-gym-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
    help='A unique job identifier.'
  )
  parser.add_argument(
    '--num_workers', default=20, type=int, help='The number of workers.'
  )
  parser.add_argument('--cloud', default=False, action='store_true')
  parser.add_argument(
    '--runner',
    help='See Dataflow runners, may be blocking'
    ' or not, on cloud or not, etc.'
  )
  parser.add_argument(
    '--image_width',
    type=int,
    required=True,
    help='Resize images to the specified width'
  )
  parser.add_argument(
    '--image_height',
    type=int,
    required=True,
    help='Resize images to the specified height'
  )
  parser.add_argument(
    '--color_map',
    type=str,
    help='The path to the color map configuration.'
  )

  parsed_args, _ = parser.parse_known_args(argv)

  if parsed_args.cloud:
    # Flags which need to be set for cloud runs.
    default_values = {
      'project':
        get_cloud_project(),
      'temp_location':
        os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
      'runner':
        'DataflowRunner',
      'save_main_session':
        True,
    }
  else:
    # Flags which need to be set for local runs.
    default_values = {
      'runner': 'DirectRunner',
    }

  for kk, vv in default_values.iteritems():
    if kk not in parsed_args or not vars(parsed_args)[kk]:
      vars(parsed_args)[kk] = vv

  return parsed_args

def main(argv):
  arg_dict = default_args(argv)
  run(arg_dict)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  main(sys.argv[1:])

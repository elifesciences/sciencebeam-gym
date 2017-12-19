from __future__ import print_function
from __future__ import absolute_import
import logging
import argparse
import io
import re

from six.moves.configparser import ConfigParser

from sciencebeam_gym.utils.tf import FileIO

def get_args_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--color_map',
    default='color_map.conf',
    type=str,
    help='The path to the color map configuration.'
  )
  parser.add_argument(
    '--input_image',
    required=True,
    type=str,
    help='The path to the input image.'
  )
  parser.add_argument(
    '--output_image',
    required=False,
    type=str,
    help='The path to the output image.'
  )
  return parser

def parse_args(argv=None):
  parser = get_args_parser()
  parsed_args, _ = parser.parse_known_args(argv)
  return parsed_args

def parse_color_map_from_configparser(color_map_config):
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
  for k, v in color_map_config.items('color_map'):
    color_map[parse_color(k)] = parse_color(v)
  return color_map

def parse_color_map_from_file(f):
  color_map_config = ConfigParser()
  color_map_config.readfp(f)
  return parse_color_map_from_configparser(color_map_config)

def parse_color_map(f):
  return parse_color_map_from_file(f)

def map_colors(img, color_map):
  if color_map is None or len(color_map) == 0:
    return img
  original_data = img.getdata()
  mapped_data = [
    color_map.get(color, color)
    for color in original_data
  ]
  img.putdata(mapped_data)
  return img

def main():
  from PIL import Image

  logger = logging.getLogger(__name__)
  args = parse_args()

  with FileIO(args.color_map, 'r') as config_f:
    color_map = parse_color_map(config_f)

  logger.info('read {} color mappings'.format(len(color_map)))

  with FileIO(args.input_image, 'rb') as input_f:
    image_bytes = input_f.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    img = map_colors(img, color_map)

    with FileIO(args.output_image, 'wb') as output_f:
      img.save(output_f, 'png')

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  main()

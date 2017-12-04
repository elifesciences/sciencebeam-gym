from __future__ import absolute_import
import re

from six import string_types
from six.moves.configparser import ConfigParser

def parse_color_map_from_configparser(color_map_config):
  num_pattern = re.compile(r'(\d+)')
  rgb_pattern = re.compile(r'\((\d+),\s*(\d+),\s*(\d+)\)')

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
    color_map[k] = parse_color(v)
  return color_map

def parse_color_map_from_file(f):
  color_map_config = ConfigParser()
  if isinstance(f, string_types):
    with open(f, 'r') as fp:
      color_map_config.readfp(fp)
  else:
    color_map_config.readfp(f)
  return parse_color_map_from_configparser(color_map_config)

def parse_color_map(f):
  return parse_color_map_from_file(f)

from __future__ import print_function
from __future__ import absolute_import
import argparse
import io

from six.moves.configparser import ConfigParser

from PIL import Image

from sciencebeam_gym.trainer.util import FileIO

def get_args_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_width',
      type=int,
      required=True,
      help='Resize images to the specified width')
  parser.add_argument(
      '--image_height',
      type=int,
      required=True,
      help='Resize images to the specified height')
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

def image_resize_bicubic(image, size):
  return image.resize(size, Image.BICUBIC)

def main():
  args = parse_args()

  image_size = (args.image_width, args.image_height)

  with FileIO(args.input_image, 'rb') as input_f:
    image_bytes = input_f.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    image = image_resize_bicubic(image, image_size)

    with FileIO(args.output_image, 'wb') as output_f:
      image.save(output_f, 'png')

if __name__ == "__main__":
  main()

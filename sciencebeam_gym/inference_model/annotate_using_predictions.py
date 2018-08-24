from __future__ import division

import argparse
import logging
from io import BytesIO

import numpy as np
from PIL import Image

from sciencebeam_utils.beam_utils.io import (
  read_all_from_path
)

from sciencebeam_gym.utils.bounding_box import (
  BoundingBox
)

from sciencebeam_gym.preprocess.color_map import (
  parse_color_map_from_file
)

from sciencebeam_gym.structured_document.structured_document_loader import (
  load_structured_document
)

from sciencebeam_gym.structured_document.structured_document_saver import (
  save_structured_document
)

CV_TAG_SCOPE = 'cv'

def get_logger():
  return logging.getLogger(__name__)

class AnnotatedImage(object):
  def __init__(self, data, color_map):
    self.data = data
    self.color_map = color_map
    self.size = (data.shape[1], data.shape[0])

  def get_tag_probabilities_within(self, bounding_box):
    image_area = self.data[
      int(bounding_box.y):int(bounding_box.y + bounding_box.height),
      int(bounding_box.x):int(bounding_box.x + bounding_box.width)
    ]
    counts = {
      k: np.sum(np.all(image_area == v, axis=-1))
      for k, v in self.color_map.items()
    }
    total = image_area.size / image_area.shape[-1]
    return {
      k: v / total if total > 0.0 else 0.0
      for k, v in counts.items()
    }

def calculate_rescale_factors(structured_document, page, annotated_image):
  page_bounding_box = structured_document.get_bounding_box(page)
  get_logger().debug('page_bounding_box: %s', page_bounding_box)
  assert page_bounding_box is not None
  page_width = page_bounding_box.width
  page_height = page_bounding_box.height
  annotated_image_width, annotated_image_height = annotated_image.size
  get_logger().debug(
    'annotated_image width, height: %f, %f',
    annotated_image_width, annotated_image_height
  )
  rx = annotated_image_width / page_width
  ry = annotated_image_height / page_height
  return rx, ry

def scale_bounding_box(bounding_box, rx, ry):
  return BoundingBox(
    bounding_box.x * rx,
    bounding_box.y * ry,
    bounding_box.width * rx,
    bounding_box.height * ry
  )

def annotate_page_using_predicted_image(
  structured_document, page, annotated_image, tag_scope=CV_TAG_SCOPE):

  rx, ry = calculate_rescale_factors(structured_document, page, annotated_image)
  get_logger().debug('rx, ry: %f, %f', rx, ry)
  for line in structured_document.get_lines_of_page(page):
    for token in structured_document.get_tokens_of_line(line):
      bounding_box = structured_document.get_bounding_box(token)
      if bounding_box:
        get_logger().debug('original bounding_box: %s', bounding_box)
        bounding_box = scale_bounding_box(bounding_box, rx, ry)
        get_logger().debug('scaled bounding_box: %s', bounding_box)
        tag_probabilites = sorted(
          ((k, v) for k, v in annotated_image.get_tag_probabilities_within(bounding_box).items()),
          key=lambda x: x[1],
          reverse=True
        )
        top_probability_tag, top_probability_value = tag_probabilites[0]
        get_logger().debug('tag_probabilites: %s', tag_probabilites)
        if top_probability_value > 0.5:
          get_logger().debug(
            'tagging token: %s: %s',
            top_probability_tag,
            structured_document.get_text(token)
          )
          structured_document.set_tag(token, top_probability_tag, scope=tag_scope)

def annotate_structured_document_using_predicted_images(
  structured_document, annotated_images, tag_scope=CV_TAG_SCOPE):

  for page, annotated_image in zip(structured_document.get_pages(), annotated_images):
    annotate_page_using_predicted_image(
      structured_document, page, annotated_image, tag_scope=tag_scope
    )
  return structured_document

def parse_args(argv=None):
  parser = argparse.ArgumentParser('Annotated LXML using prediction images')
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument(
    '--lxml-path', type=str, required=False,
    help='path to lxml or svg pages document'
  )

  images = parser.add_mutually_exclusive_group(required=True)
  images.add_argument(
    '--images-path', type=str, nargs='+',
    help='path to lxml document'
  )

  parser.add_argument(
    '--output-path', type=str, required=True,
    help='output path to annotated document'
  )

  parser.add_argument(
    '--tag-scope', type=str, required=False,
    default=CV_TAG_SCOPE,
    help='target tag scope for the predicted tags'
  )

  parser.add_argument(
    '--color-map', default='color_map.conf',
    help='color map to use'
  )

  parser.add_argument(
    '--debug', action='store_true', default=False,
    help='enable debug output'
  )

  return parser.parse_args(argv)

def load_annotation_image(path, color_map):
  get_logger().debug('loading annotation image: %s', path)
  return AnnotatedImage(
    np.asarray(
      Image.open(BytesIO(read_all_from_path(path, 'rb'))).convert('RGB'),
      dtype=np.uint8
    ),
    color_map
  )

def main(argv=None):
  args = parse_args(argv)

  if args.debug:
    logging.getLogger().setLevel('DEBUG')

  color_map = parse_color_map_from_file(args.color_map)
  get_logger().debug('color_map: %s', color_map)

  structured_document = load_structured_document(args.lxml_path, 'rb')

  annotated_images = (
    load_annotation_image(path, color_map)
    for path in args.images_path
  )

  structured_document = annotate_structured_document_using_predicted_images(
    structured_document,
    annotated_images,
    tag_scope=args.tag_scope
  )

  get_logger().info('writing result to: %s', args.output_path)
  save_structured_document(args.output_path, structured_document.root)

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

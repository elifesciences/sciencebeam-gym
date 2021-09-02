import argparse
import json
import logging
import os
from io import BytesIO
from typing import Iterable, List, Optional

import PIL.Image
from lxml import etree
from pdf2image import convert_from_bytes

from sciencebeam_gym.utils.io import read_bytes, write_text
from sciencebeam_gym.utils.image_object_matching import (
    get_bounding_box_for_image,
    get_image_list_object_match,
    get_sift_detector_matcher
)


LOGGER = logging.getLogger(__name__)


XLINK_NS = 'http://www.w3.org/1999/xlink'
XLINK_NS_PREFIX = '{%s}' % XLINK_NS
XLINK_HREF = XLINK_NS_PREFIX + 'href'


def get_images_from_pdf(pdf_path: str) -> List[PIL.Image.Image]:
    return convert_from_bytes(read_bytes(pdf_path))


def iter_graphic_element_hrefs_from_xml_node(
    xml_root: etree.ElementBase
) -> Iterable[str]:
    for graphic_element in xml_root.xpath('//graphic'):
        href = graphic_element.attrib.get(XLINK_HREF)
        if href:
            yield href


def get_graphic_element_paths_from_xml_file(
    xml_path: str
) -> List[str]:
    xml_dirname = os.path.dirname(xml_path)
    return [
        os.path.join(xml_dirname, href)
        for href in iter_graphic_element_hrefs_from_xml_node(
            etree.fromstring(read_bytes(xml_path))
        )
    ]


def read_bytes_with_optional_gz_extension(path_or_url: str) -> bytes:
    if not path_or_url.endswith('.gz'):
        try:
            return read_bytes(path_or_url + '.gz')
        except FileNotFoundError:
            LOGGER.debug(
                'file not found %r, attempting to read %r',
                path_or_url + '.gz', path_or_url
            )
    return read_bytes(path_or_url)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pdf-file',
        type=str,
        required=True,
        help='Path to the PDF file'
    )
    xml_image_group = parser.add_mutually_exclusive_group(required=True)
    xml_image_group.add_argument(
        '--image-files',
        nargs='+',
        type=str,
        help='Path to the images to find the bounding boxes for'
    )
    xml_image_group.add_argument(
        '--xml-file',
        type=str,
        help='Path to the xml file, whoes graphic elements to find the bounding boxes for'
    )
    parser.add_argument(
        '--output-json-file',
        required=True,
        type=str,
        help='The path to the output JSON file to write the bounding boxes to.'
    )
    return parser


def parse_args(argv: Optional[List[str]] = None):
    parser = get_args_parser()
    parsed_args, _ = parser.parse_known_args(argv)
    return parsed_args


def run(
    pdf_path: str,
    image_paths: Optional[List[str]],
    xml_path: Optional[str],
    json_path: str
):
    pdf_images = get_images_from_pdf(pdf_path)
    if xml_path:
        image_paths = get_graphic_element_paths_from_xml_file(xml_path)
    else:
        assert image_paths is not None
    object_detector_matcher = get_sift_detector_matcher()
    annotations = []
    for image_path in image_paths:
        template_image = PIL.Image.open(BytesIO(read_bytes_with_optional_gz_extension(
            image_path
        )))
        LOGGER.debug('template_image: %s x %s', template_image.width, template_image.height)
        image_list_match_result = get_image_list_object_match(
            pdf_images,
            template_image,
            object_detector_matcher=object_detector_matcher
        )
        if not image_list_match_result:
            continue
        page_index = image_list_match_result.target_image_index
        pdf_image = pdf_images[page_index]
        pdf_page_bounding_box = get_bounding_box_for_image(pdf_image)
        bounding_box = image_list_match_result.target_bounding_box
        if bounding_box:
            LOGGER.debug('bounding_box: %s', bounding_box)
            annotations.append({
                'image_id': (1 + page_index),
                'category_id': 1,
                'bbox': bounding_box.intersection(pdf_page_bounding_box).to_list()
            })
    data_json = {
        'images': [
            {
                'file_name': os.path.basename(pdf_path) + '/page_%05d.jpg' % (1 + page_index),
                'width': pdf_image.width,
                'height': pdf_image.height,
                'id': (1 + page_index)
            }
            for page_index, pdf_image in enumerate(pdf_images)
        ],
        'annotations': annotations,
        'categories': [{
            'id': 1,
            'name': 'figure'
        }]
    }
    write_text(json_path, json.dumps(data_json, indent=2))


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    LOGGER.info('args: %s', args)
    run(
        pdf_path=args.pdf_file,
        image_paths=args.image_files,
        xml_path=args.xml_file,
        json_path=args.output_json_file
    )


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()

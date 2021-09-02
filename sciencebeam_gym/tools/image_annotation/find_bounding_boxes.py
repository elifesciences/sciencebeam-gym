import argparse
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, NamedTuple, Optional

import PIL.Image
from lxml import etree
from pdf2image import convert_from_bytes

from sciencebeam_utils.utils.progress_logger import logging_tqdm

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


class CategoryNames:
    FIGURE = 'figure'
    FORMULA = 'formula'
    TABLE = 'table'
    UNKNOWN_GRAPHIC = 'unknown_graphic'


class GraphicImageDescriptor(NamedTuple):
    href: str
    path: str
    category_name: str
    related_element_id: Optional[str] = None


CATEGROY_NAME_BY_XML_TAG = {
    'disp-formula': CategoryNames.FORMULA,
    'fig': CategoryNames.FIGURE,
    'table-wrap': CategoryNames.TABLE
}


def get_category_name_by_xml_node(xml_node: etree.ElementBase) -> str:
    while xml_node is not None:
        category_name = CATEGROY_NAME_BY_XML_TAG.get(xml_node.tag)
        if category_name:
            return category_name
        xml_node = xml_node.getparent()
    return CategoryNames.UNKNOWN_GRAPHIC


def get_related_element_id_by_xml_node(xml_node: etree.ElementBase) -> Optional[str]:
    while xml_node is not None:
        related_element_id = xml_node.attrib.get('id')
        if related_element_id:
            return related_element_id
        xml_node = xml_node.getparent()
    return None


def iter_graphic_element_descriptors_from_xml_node(
    xml_root: etree.ElementBase,
    parent_dirname: str
) -> Iterable[GraphicImageDescriptor]:
    for graphic_element in xml_root.xpath('//graphic'):
        href = graphic_element.attrib.get(XLINK_HREF)
        if href:
            yield GraphicImageDescriptor(
                href=href,
                path=os.path.join(parent_dirname, href),
                category_name=get_category_name_by_xml_node(graphic_element),
                related_element_id=get_related_element_id_by_xml_node(graphic_element)
            )


def get_graphic_element_descriptors_from_xml_file(
    xml_path: str
) -> List[GraphicImageDescriptor]:
    return list(iter_graphic_element_descriptors_from_xml_node(
        etree.fromstring(read_bytes(xml_path)),
        parent_dirname=os.path.dirname(xml_path)
    ))


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
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
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
    parser.add_argument(
        '--max-internal-width',
        type=int,
        default=1280,
        help='Maximum internal width (for faster processing)'
    )
    parser.add_argument(
        '--max-internal-height',
        type=int,
        default=1280,
        help='Maximum internal height (for faster processing)'
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
    json_path: str,
    max_internal_width: int,
    max_internal_height: int
):
    pdf_images = get_images_from_pdf(pdf_path)
    if xml_path:
        image_descriptors = get_graphic_element_descriptors_from_xml_file(xml_path)
    else:
        assert image_paths is not None
        image_descriptors = [
            GraphicImageDescriptor(
                href=image_path,
                path=image_path,
                category_name=CategoryNames.UNKNOWN_GRAPHIC
            )
            for image_path in image_paths
        ]
    object_detector_matcher = get_sift_detector_matcher()
    category_id_by_name: Dict[str, int] = {}
    annotations = []
    image_cache: Dict[Any, Any] = {}
    for image_descriptor in logging_tqdm(
        image_descriptors,
        logger=LOGGER,
        desc='processing images:'
    ):
        template_image = PIL.Image.open(BytesIO(read_bytes_with_optional_gz_extension(
            image_descriptor.path
        )))
        LOGGER.debug('template_image: %s x %s', template_image.width, template_image.height)
        image_list_match_result = get_image_list_object_match(
            pdf_images,
            template_image,
            object_detector_matcher=object_detector_matcher,
            image_cache=image_cache,
            max_width=max_internal_width,
            max_height=max_internal_height
        )
        if not image_list_match_result:
            continue
        page_index = image_list_match_result.target_image_index
        pdf_image = pdf_images[page_index]
        pdf_page_bounding_box = get_bounding_box_for_image(pdf_image)
        bounding_box = image_list_match_result.target_bounding_box
        if bounding_box:
            LOGGER.debug('bounding_box: %s', bounding_box)
            category_id = category_id_by_name.get(image_descriptor.category_name)
            if category_id is None:
                category_id = 1 + len(category_id_by_name)
                category_id_by_name[image_descriptor.category_name] = category_id
            annotation = {
                'image_id': (1 + page_index),
                'file_name': image_descriptor.href,
                'category_id': category_id,
                'bbox': bounding_box.intersection(pdf_page_bounding_box).to_list()
            }
            if image_descriptor.related_element_id:
                annotation['related_element_id'] = image_descriptor.related_element_id
            annotations.append(annotation)
    data_json = {
        'info': {
            'version': '0.0.1',
            'date_created': datetime.utcnow().isoformat()
        },
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
        'categories': [
            {
                'id': category_id,
                'name': category_name
            }
            for category_name, category_id in category_id_by_name.items()
        ]
    }
    write_text(json_path, json.dumps(data_json, indent=2))


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.debug:
        logging.getLogger('sciencebeam_gym').setLevel(logging.DEBUG)
    LOGGER.info('args: %s', args)
    run(
        pdf_path=args.pdf_file,
        image_paths=args.image_files,
        xml_path=args.xml_file,
        json_path=args.output_json_file,
        max_internal_width=args.max_internal_width,
        max_internal_height=args.max_internal_height,
    )


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()

# moved most functionallity out of main due to:
#   https://issues.apache.org/jira/browse/BEAM-6158
import argparse
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, cast

import matplotlib.cm
import PIL.Image
import numpy as np
from lxml import etree
from pdf2image import convert_from_bytes

from sciencebeam_utils.utils.file_path import relative_path
from sciencebeam_utils.utils.progress_logger import logging_tqdm
from sciencebeam_utils.utils.file_list import load_file_list

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.utils.collections import get_inverted_dict
from sciencebeam_gym.utils.io import read_bytes, write_bytes, write_text
from sciencebeam_gym.utils.image_object_matching import (
    DEFAULT_MAX_BOUNDING_BOX_ADJUSTMENT_ITERATIONS,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MAX_WIDTH,
    get_bounding_box_for_image,
    get_image_list_object_match,
    get_sift_detector_matcher
)
from sciencebeam_gym.utils.visualize_bounding_box import draw_bounding_box
from sciencebeam_gym.utils.pipeline import (
    AbstractPipelineFactory,
    add_pipeline_args,
    process_pipeline_args
)


LOGGER = logging.getLogger(__name__)


XLINK_NS = 'http://www.w3.org/1999/xlink'
XLINK_NS_PREFIX = '{%s}' % XLINK_NS
XLINK_HREF = XLINK_NS_PREFIX + 'href'


COORDS_ATTRIB_NAME = 'coords'


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
    element: Optional[etree.ElementBase] = None


class GraphicImageNotFoundError(RuntimeError):
    pass


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
                related_element_id=get_related_element_id_by_xml_node(graphic_element),
                element=graphic_element
            )


def get_graphic_element_descriptors_from_xml_node(
    *args, **kwargs
) -> List[GraphicImageDescriptor]:
    return list(iter_graphic_element_descriptors_from_xml_node(
        *args, **kwargs
    ))


def get_graphic_element_descriptors_from_xml_file(
    xml_path: str
) -> List[GraphicImageDescriptor]:
    return get_graphic_element_descriptors_from_xml_node(
        etree.fromstring(read_bytes(xml_path)),
        parent_dirname=os.path.dirname(xml_path)
    )


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
    pdf_file_group = parser.add_mutually_exclusive_group(required=True)
    pdf_file_group.add_argument(
        '--pdf-file-list',
        type=str,
        help='Path to the PDF file list'
    )
    pdf_file_group.add_argument(
        '--pdf-file',
        type=str,
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
        '--xml-file-list',
        type=str,
        help='Path to the xml file list, whoes graphic elements to find the bounding boxes for'
    )
    xml_image_group.add_argument(
        '--xml-file',
        type=str,
        help='Path to the xml file, whoes graphic elements to find the bounding boxes for'
    )
    parser.add_argument(
        '--pdf-base-path',
        type=str,
        help=(
            'The PDF base path is used to determine the output directory'
            ' based on the source folder.'
            ' This results in sub directories in --output-path,'
            ' if the source file is also in a sub directory.'
        )
    )
    parser.add_argument(
        '--pdf-file-column',
        type=str,
        default='source_url',
        help='The column for --pdf-file-list (if tsv or csv).'
    )
    parser.add_argument(
        '--xml-file-column',
        type=str,
        default='xml_url',
        help='The column for --xml-file-list (if tsv or csv).'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help=(
            'The limit argument allows you to limit the number of documents to process,'
            ' when using file lists.'
        )
    )
    parser.add_argument(
        '--output-path',
        type=str,
        help='The base output path to write files to (required for file lists).'
    )
    parser.add_argument(
        '--output-json-file',
        required=True,
        type=str,
        help='The path to the output JSON file to write the bounding boxes to.'
    )
    parser.add_argument(
        '--output-xml-file',
        type=str,
        help=(
            'The path to the output XML file to write the bounding boxes to.'
            ' This will be the original XML with bounding box added to it.'
        )
    )
    parser.add_argument(
        '--output-annotated-images-path',
        required=False,
        type=str,
        help=(
            'The path to the output directory, that annotated images should be saved to.'
            ' Disabled, if not specified.'
        )
    )
    parser.add_argument(
        '--max-internal-width',
        type=int,
        default=DEFAULT_MAX_WIDTH,
        help='Maximum internal width (for faster processing)'
    )
    parser.add_argument(
        '--max-internal-height',
        type=int,
        default=DEFAULT_MAX_HEIGHT,
        help='Maximum internal height (for faster processing)'
    )
    parser.add_argument(
        '--use-grayscale',
        action='store_true',
        help='Convert images to grayscale internally'
    )
    parser.add_argument(
        '--skip-errors',
        action='store_true',
        help='Skip errors finding bounding boxes and output missing annotations'
    )
    parser.add_argument(
        '--max-bounding-box-adjustment-iterations',
        type=int,
        default=DEFAULT_MAX_BOUNDING_BOX_ADJUSTMENT_ITERATIONS,
        help=(
            'Maximum bounding box adjustment iterations (0 to disable).'
            ' Sometimes the bounding box returned by the algorithm is slightly off.'
            ' With bounding box adjustments, the final bounding box are adjusted'
            ' in order to maximise the score.'
        )
    )
    add_pipeline_args(parser)
    return parser


def process_args(args: argparse.Namespace):
    process_pipeline_args(args, args.output_path)
    if args.pdf_file_list or args.xml_file_list:
        if not args.pdf_file_list or not args.xml_file_list:
            raise RuntimeError(
                'both --pdf-file-list and -xml-file-list must be used together'
            )
    if args.pdf_file_list and args.image_files:
        raise RuntimeError('--images-files cannot be used together with --pdf-file-list')
    if args.pdf_file_list and not args.pdf_base_path:
        raise RuntimeError('--pdf-base-path required for --pdf-file-list')
    if args.output_xml_file and not (args.xml_file_list or args.xml_file):
        raise RuntimeError('--xml-file or --xml-file-list required for --output-xml-file')


def parse_args(argv: Optional[List[str]] = None):
    parser = get_args_parser()
    parsed_args = parser.parse_args(argv)
    return parsed_args


def save_annotated_images(
    pdf_images: List[PIL.Image.Image],
    annotations: List[dict],
    output_annotated_images_path: str,
    category_name_by_id: Dict[int, str]
):
    cmap = matplotlib.cm.get_cmap('Set1')
    for page_index, page_image in enumerate(pdf_images):
        page_image_id = (1 + page_index)
        output_filename = 'page_%05d.png' % page_image_id
        full_output_path = os.path.join(output_annotated_images_path, output_filename)
        page_annotations = [
            annotation
            for annotation in annotations
            if annotation['image_id'] == page_image_id
        ]
        page_image_array = np.copy(np.asarray(page_image))
        for annotation in page_annotations:
            category_name = category_name_by_id[annotation['category_id']]
            bounding_box = BoundingBox(*annotation['bbox']).round()
            color: Tuple[int, int, int] = cast(Tuple[int, int, int], tuple((
                int(v)
                for v in (
                    np.asarray(cmap(annotation['category_id'])[:3]) * 255
                )
            )))
            related_element_id = annotation.get('related_element_id')
            score = annotation.get('_score')
            text = f'{category_name}: {annotation["file_name"]}'
            if related_element_id:
                text += f' ({related_element_id})'
            if score is not None:
                text += ' (%.2f)' % score
            draw_bounding_box(
                page_image_array,
                bounding_box=bounding_box,
                color=color,
                text=text
            )
        image_png_bio = BytesIO()
        PIL.Image.fromarray(page_image_array).save(image_png_bio, format='PNG')
        write_bytes(full_output_path, image_png_bio.getvalue())


def format_coords_attribute_value(
    page_number: int,
    bounding_box: BoundingBox
) -> str:
    return ','.join([
        str(v)
        for v in [page_number] + bounding_box.to_list()
    ])


def process_single_document(
    pdf_path: str,
    image_paths: Optional[List[str]],
    xml_path: Optional[str],
    output_json_path: str,
    max_internal_width: int,
    max_internal_height: int,
    use_grayscale: bool,
    skip_errors: bool,
    max_bounding_box_adjustment_iterations: int,
    output_xml_path: Optional[str] = None,
    output_annotated_images_path: Optional[str] = None
):
    pdf_images = get_images_from_pdf(pdf_path)
    xml_root: Optional[etree.ElementBase] = None
    if xml_path:
        xml_root = etree.fromstring(read_bytes(xml_path))
        image_descriptors = get_graphic_element_descriptors_from_xml_node(
            xml_root,
            parent_dirname=os.path.dirname(xml_path)
        )
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
    annotations: List[dict] = []
    missing_annotations: List[dict] = []
    image_cache: Dict[Any, Any] = {}
    for image_descriptor in logging_tqdm(
        image_descriptors,
        logger=LOGGER,
        desc='processing images(%r):' % os.path.basename(pdf_path)
    ):
        LOGGER.debug('processing article image: %r', image_descriptor.href)
        template_image = PIL.Image.open(BytesIO(read_bytes_with_optional_gz_extension(
            image_descriptor.path
        )))
        LOGGER.debug('template_image: %s x %s', template_image.width, template_image.height)
        image_list_match_result = get_image_list_object_match(
            pdf_images,
            template_image,
            object_detector_matcher=object_detector_matcher,
            image_cache=image_cache,
            template_image_id=f'{id(image_descriptor)}-{image_descriptor.href}',
            max_width=max_internal_width,
            max_height=max_internal_height,
            use_grayscale=use_grayscale,
            max_bounding_box_adjustment_iterations=max_bounding_box_adjustment_iterations
        )
        category_id = category_id_by_name.get(image_descriptor.category_name)
        if category_id is None:
            category_id = 1 + len(category_id_by_name)
            category_id_by_name[image_descriptor.category_name] = category_id
        annotation = {
            'file_name': image_descriptor.href,
            'category_id': category_id
        }
        if image_descriptor.related_element_id:
            annotation['related_element_id'] = image_descriptor.related_element_id
        if not image_list_match_result:
            if not skip_errors:
                raise GraphicImageNotFoundError(
                    'image bounding box not found for: %r' % image_descriptor.href
                )
            missing_annotations.append(annotation)
            continue
        page_index = image_list_match_result.target_image_index
        pdf_image = pdf_images[page_index]
        pdf_page_bounding_box = get_bounding_box_for_image(pdf_image)
        bounding_box = image_list_match_result.target_bounding_box
        assert bounding_box
        LOGGER.debug('bounding_box: %s', bounding_box)
        normalized_bounding_box = bounding_box.intersection(pdf_page_bounding_box).round()
        annotation = {
            **annotation,
            'image_id': (1 + page_index),
            'bbox': normalized_bounding_box.to_list(),
            '_score': image_list_match_result.score
        }
        annotations.append(annotation)
        if image_descriptor.element is not None:
            image_descriptor.element.attrib[COORDS_ATTRIB_NAME] = (
                format_coords_attribute_value(
                    page_number=1 + page_index,
                    bounding_box=normalized_bounding_box
                )
            )
    if output_annotated_images_path:
        LOGGER.info('saving annotated images to: %r', output_annotated_images_path)
        save_annotated_images(
            pdf_images=pdf_images,
            annotations=annotations,
            output_annotated_images_path=output_annotated_images_path,
            category_name_by_id=get_inverted_dict(category_id_by_name)
        )
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
    if missing_annotations:
        data_json['missing_annotations'] = missing_annotations
    LOGGER.info('writing to: %r', output_json_path)
    write_text(output_json_path, json.dumps(data_json, indent=2))
    if output_xml_path and xml_root is not None:
        LOGGER.info('writing to: %r', output_xml_path)
        write_bytes(output_xml_path, etree.tostring(xml_root))


class FindBoundingBoxItem(NamedTuple):
    pdf_file: str
    xml_file: str
    image_files: Optional[List[str]] = None


class FindBoundingBoxPipelineFactory(AbstractPipelineFactory[FindBoundingBoxItem]):
    def __init__(self, args: argparse.Namespace):
        super().__init__(resume=args.resume)
        self.args = args
        self.max_internal_width = args.max_internal_width
        self.max_internal_height = args.max_internal_height
        self.use_grayscale = args.use_grayscale
        self.skip_errors = args.skip_errors
        self.max_bounding_box_adjustment_iterations = args.max_bounding_box_adjustment_iterations

    def process_item(self, item: FindBoundingBoxItem):
        output_json_file = self.get_output_file_for_item(item)
        output_xml_file = self.get_output_xml_file_for_item(item)
        output_annotated_images_path = self.get_output_annotated_images_directory_for_item(item)
        process_single_document(
            pdf_path=item.pdf_file,
            image_paths=item.image_files,
            xml_path=item.xml_file,
            output_json_path=output_json_file,
            max_internal_width=self.max_internal_width,
            max_internal_height=self.max_internal_height,
            use_grayscale=self.use_grayscale,
            skip_errors=self.skip_errors,
            output_xml_path=output_xml_file,
            output_annotated_images_path=output_annotated_images_path,
            max_bounding_box_adjustment_iterations=self.max_bounding_box_adjustment_iterations
        )

    def get_item_list(self):
        args = self.args
        pdf_file_list: List[str]
        xml_file_list: List[str]
        image_files: Optional[List[str]] = None
        if args.pdf_file_list:
            assert args.xml_file_list
            pdf_file_list = load_file_list(
                args.pdf_file_list, column=args.pdf_file_column, limit=args.limit
            )
            xml_file_list = load_file_list(
                args.xml_file_list, column=args.xml_file_column, limit=args.limit
            )
        else:
            pdf_file_list = [args.pdf_file]
            xml_file_list = [args.xml_file]
            image_files = args.image_files
        assert len(pdf_file_list) == len(xml_file_list), \
            f'number of pdf and xml files must match: {len(pdf_file_list)} != {len(xml_file_list)}'
        LOGGER.debug('processing: pdf_file_list=%r, xml_file_list=%r', pdf_file_list, xml_file_list)
        return [
            FindBoundingBoxItem(
                pdf_file=pdf_file,
                xml_file=xml_file,
                image_files=image_files
            )
            for pdf_file, xml_file in zip(pdf_file_list, xml_file_list)
        ]

    def get_output_directory_for_item(self, item: FindBoundingBoxItem) -> str:
        if self.args.output_path and self.args.pdf_base_path:
            return os.path.join(
                self.args.output_path,
                relative_path(
                    self.args.pdf_base_path,
                    os.path.dirname(item.pdf_file)
                )
            )
        return self.args.output_path

    def get_output_json_file_for_item(self, item: FindBoundingBoxItem) -> str:
        output_path = self.get_output_directory_for_item(item)
        if output_path:
            return os.path.join(
                output_path,
                self.args.output_json_file
            )
        return self.args.output_json_file

    def get_output_xml_file_for_item(self, item: FindBoundingBoxItem) -> str:
        output_path = self.get_output_directory_for_item(item)
        if output_path:
            return os.path.join(
                output_path,
                self.args.output_xml_file
            )
        return self.args.output_xml_file

    def get_output_annotated_images_directory_for_item(self, item: FindBoundingBoxItem) -> str:
        output_path = self.get_output_directory_for_item(item)
        if output_path:
            return os.path.join(
                output_path,
                self.args.output_annotated_images_path
            )
        return self.args.output_annotated_images_path

    def get_output_file_for_item(self, item: FindBoundingBoxItem) -> str:
        return self.get_output_json_file_for_item(item)


def run(args: argparse.Namespace):
    FindBoundingBoxPipelineFactory(args).run(
        args
    )


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.debug:
        for name in ['__main__', 'sciencebeam_gym']:
            logging.getLogger(name).setLevel(logging.DEBUG)
    LOGGER.info('args: %s', args)
    process_args(args)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()

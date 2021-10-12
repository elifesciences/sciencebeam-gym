# moved most functionallity out of main due to:
#   https://issues.apache.org/jira/browse/BEAM-6158
import argparse
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, cast

import cachetools
import diskcache
import matplotlib.cm
import PIL.Image
import pdf2image
import numpy as np
from lxml import etree

from sciencebeam_utils.utils.string import parse_list
from sciencebeam_utils.utils.file_path import get_output_file
from sciencebeam_utils.utils.progress_logger import logging_tqdm
from sciencebeam_utils.utils.file_list import load_file_list

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.utils.cache import MultiLevelCache
from sciencebeam_gym.utils.collections import get_inverted_dict
from sciencebeam_gym.utils.cv import load_pil_image_from_file
from sciencebeam_gym.utils.io import copy_file, read_bytes, write_bytes, write_text
from sciencebeam_gym.utils.image_object_matching import (
    DEFAULT_MAX_BOUNDING_BOX_ADJUSTMENT_ITERATIONS,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MAX_WIDTH,
    EMPTY_IMAGE_LIST_OBJECT_MATCH_RESULT,
    get_bounding_box_for_image,
    get_sift_detector_matcher,
    iter_current_best_image_list_object_match
)
from sciencebeam_gym.utils.pickle_reg import register_pickle_functions
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

COORDS_NS = 'http://www.tei-c.org/ns/1.0'
COORDS_NS_PREFIX = '{%s}' % COORDS_NS
COORDS_NS_NAMEMAP = {'grobid-tei': COORDS_NS}
COORDS_ATTRIB_NAME = COORDS_NS_PREFIX + 'coords'

DEFAULT_OUTPUT_JSON_FILE_SUFFIX = '.annotation.coco.json'
DEFAULT_OUTPUT_XML_FILE_SUFFIX = '.annotated.xml'
DEFAULT_OUTPUT_ANNOTATED_IMAGES_DIR__SUFFIX = '-annotated-images'

DEFAULT_MEMORY_CACHE_SIZE = 64


def get_images_from_pdf(pdf_path: str, pdf_scale_to: Optional[int]) -> List[PIL.Image.Image]:
    with TemporaryDirectory(suffix='-pdf') as temp_dir:
        local_pdf_path = os.path.join(temp_dir, os.path.basename(pdf_path))
        if local_pdf_path.endswith('.gz'):
            local_pdf_path, _ = os.path.splitext(local_pdf_path)
        LOGGER.debug('copying PDF file from %r to %r', pdf_path, local_pdf_path)
        copy_file(pdf_path, local_pdf_path)
        file_size = os.path.getsize(local_pdf_path)
        LOGGER.info(
            'rendering PDF file (%d bytes, scale to: %r): %r',
            file_size, pdf_scale_to, pdf_path
        )
        pdf_image_paths = pdf2image.convert_from_path(
            local_pdf_path,
            paths_only=True,
            output_folder=temp_dir,
            size=pdf_scale_to
        )
        pdf_image_paths = logging_tqdm(
            pdf_image_paths,
            logger=LOGGER,
            desc='loading PDF image(%r):' % os.path.basename(pdf_path)
        )
        pdf_images = [
            load_pil_image_from_file(pdf_image_path)
            for pdf_image_path in pdf_image_paths
        ]
        LOGGER.info('loaded rendered PDF images(%r)', os.path.basename(pdf_path))
        return pdf_images


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
        required=True,
        help='The base output path to write files to (required for file lists).'
    )
    parser.add_argument(
        '--output-json-file-suffix',
        type=str,
        default=DEFAULT_OUTPUT_JSON_FILE_SUFFIX,
        help=(
            'The suffix forms part of the path to the output JSON file'
            ' to write the bounding boxes to.'
            ' The path will be <output path>/<relative sub dir>/'
            '<pdf basename without ext><output suffix>'
        )
    )
    parser.add_argument(
        '--output-xml-file-suffix',
        type=str,
        default=DEFAULT_OUTPUT_XML_FILE_SUFFIX,
        help=(
            'Part of the path to the output XML file to write the bounding boxes to.'
            ' This will be the original XML with bounding box added to it.'
            ' (requires --save-annotated-xml)'
        )
    )
    parser.add_argument(
        '--output-annotated-images-dir-suffix',
        type=str,
        default=DEFAULT_OUTPUT_ANNOTATED_IMAGES_DIR__SUFFIX,
        help=(
            'Part of the path to the output directory, that annotated images should be saved to.'
            ' (requires --save-annotated-images).'
        )
    )
    parser.add_argument(
        '--save-annotated-xml',
        action='store_true',
        help='Enable saving of annotated xml'
    )
    parser.add_argument(
        '--save-annotated-images',
        action='store_true',
        help='Enable saving of annotated images'
    )
    parser.add_argument(
        '--categories',
        type=parse_list,
        help='If specified, only process images with the specified categories (comma separated)'
    )
    parser.add_argument(
        '--pdf-scale-to',
        type=int,
        help='If specified, rendered PDF pages will be scaled to specified value (longest side)'
    )
    parser.add_argument(
        '--memory-cache-size',
        type=int,
        default=DEFAULT_MEMORY_CACHE_SIZE,
        help='Number of items to keep in the memory cache'
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
        '--ignore-unmatched-graphics',
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
    if args.save_annotated_xml and not (args.xml_file_list or args.xml_file):
        raise RuntimeError('--xml-file or --xml-file-list required for --save-annotated-xml')


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


def get_xml_root_with_update_nsmap(
    xml_root: etree.ElementBase,
    nsmap: Dict[str, str]
) -> etree.ElementBase:
    updated_root = etree.Element(xml_root.tag, nsmap=nsmap)
    updated_root[:] = xml_root[:]
    return updated_root


def get_cache(temp_dir: str, memory_cache_size: int):
    register_pickle_functions()
    LOGGER.info('using cache dir: %r (memory_cache_size: %r)', temp_dir, memory_cache_size)
    return MultiLevelCache([
        cachetools.LRUCache(maxsize=memory_cache_size),
        diskcache.Cache(directory=temp_dir)
    ])


def process_single_document(
    pdf_path: str,
    image_paths: Optional[List[str]],
    xml_path: Optional[str],
    output_json_path: str,
    pdf_scale_to: Optional[int],
    max_internal_width: int,
    max_internal_height: int,
    use_grayscale: bool,
    ignore_unmatched_graphics: bool,
    max_bounding_box_adjustment_iterations: int,
    temp_dir: str,
    memory_cache_size: int,
    selected_categories: Sequence[str] = tuple([]),
    output_xml_path: Optional[str] = None,
    output_annotated_images_path: Optional[str] = None
):
    pdf_images = get_images_from_pdf(pdf_path, pdf_scale_to=pdf_scale_to)
    xml_root: Optional[etree.ElementBase] = None
    if xml_path:
        LOGGER.info('parsing XML file(%r)', os.path.basename(xml_path))
        xml_root = etree.fromstring(read_bytes(xml_path))
        image_descriptors = get_graphic_element_descriptors_from_xml_node(
            xml_root,
            parent_dirname=os.path.dirname(xml_path)
        )
        if selected_categories:
            image_descriptors = [
                image_descriptor
                for image_descriptor in image_descriptors
                if image_descriptor.category_name in selected_categories
            ]
        LOGGER.info('updating XML namespace for file(%r)', os.path.basename(xml_path))
        xml_root = get_xml_root_with_update_nsmap(xml_root, {
            **xml_root.nsmap,
            **COORDS_NS_NAMEMAP
        })
        LOGGER.info('done parsing XML file(%r)', os.path.basename(xml_path))
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
    image_cache = get_cache(temp_dir, memory_cache_size=memory_cache_size)
    LOGGER.info(
        'start processing images(%r): %d',
        os.path.basename(pdf_path), len(image_descriptors)
    )
    with logging_tqdm(
        total=len(image_descriptors) * len(pdf_images),
        logger=LOGGER,
        desc='processing images(%r):' % os.path.basename(pdf_path)
    ) as pbar:
        for image_descriptor in image_descriptors:
            LOGGER.debug('processing article image: %r', image_descriptor.href)
            template_image = PIL.Image.open(BytesIO(read_bytes_with_optional_gz_extension(
                image_descriptor.path
            )))
            LOGGER.debug('template_image: %s x %s', template_image.width, template_image.height)
            image_list_match_result = EMPTY_IMAGE_LIST_OBJECT_MATCH_RESULT
            for _image_list_match_result in iter_current_best_image_list_object_match(
                pdf_images,
                template_image,
                object_detector_matcher=object_detector_matcher,
                image_cache=image_cache,
                template_image_id=f'{id(image_descriptor)}-{image_descriptor.href}',
                max_width=max_internal_width,
                max_height=max_internal_height,
                use_grayscale=use_grayscale,
                max_bounding_box_adjustment_iterations=max_bounding_box_adjustment_iterations
            ):
                image_list_match_result = _image_list_match_result
                pbar.update(1)
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
                if not ignore_unmatched_graphics:
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
        super().__init__(
            **AbstractPipelineFactory.get_init_kwargs_for_parsed_args(args)
        )
        self.args = args
        self.output_base_path = args.output_path
        self.pdf_base_path = args.pdf_base_path
        self.output_json_file_suffix = args.output_json_file_suffix
        self.output_xml_file_suffix = args.output_xml_file_suffix
        self.output_annotated_images_dir_suffix = args.output_annotated_images_dir_suffix
        self.save_annotated_xml_enabled = args.save_annotated_xml
        self.save_annotated_images_enabled = args.save_annotated_images
        self.selected_categories = args.categories
        self.pdf_scale_to = args.pdf_scale_to
        self.memory_cache_size = args.memory_cache_size
        self.max_internal_width = args.max_internal_width
        self.max_internal_height = args.max_internal_height
        self.use_grayscale = args.use_grayscale
        self.ignore_unmatched_graphics = args.ignore_unmatched_graphics
        self.max_bounding_box_adjustment_iterations = args.max_bounding_box_adjustment_iterations

    def process_item(self, item: FindBoundingBoxItem):
        output_json_file = self.get_output_file_for_item(item)
        output_xml_file = (
            self.get_output_xml_file_for_item(item)
            if self.save_annotated_xml_enabled
            else None
        )
        output_annotated_images_path = (
            self.get_output_annotated_images_directory_for_item(item)
            if self.save_annotated_images_enabled
            else None
        )
        with TemporaryDirectory(suffix='-find-bbox') as temp_dir:
            process_single_document(
                temp_dir=temp_dir,
                pdf_path=item.pdf_file,
                image_paths=item.image_files,
                xml_path=item.xml_file,
                output_json_path=output_json_file,
                selected_categories=self.selected_categories,
                pdf_scale_to=self.pdf_scale_to,
                memory_cache_size=self.memory_cache_size,
                max_internal_width=self.max_internal_width,
                max_internal_height=self.max_internal_height,
                use_grayscale=self.use_grayscale,
                ignore_unmatched_graphics=self.ignore_unmatched_graphics,
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

    def get_output_file_or_dir_for_item(
        self,
        item: FindBoundingBoxItem,
        suffix: str
    ) -> str:
        return get_output_file(
            filename=item.pdf_file,
            source_base_path=self.pdf_base_path or os.path.dirname(item.pdf_file),
            output_base_path=self.output_base_path,
            output_file_suffix=suffix
        )

    def get_output_json_file_for_item(self, item: FindBoundingBoxItem) -> str:
        return self.get_output_file_or_dir_for_item(
            item,
            self.output_json_file_suffix
        )

    def get_output_xml_file_for_item(self, item: FindBoundingBoxItem) -> str:
        return self.get_output_file_or_dir_for_item(
            item,
            self.output_xml_file_suffix
        )

    def get_output_annotated_images_directory_for_item(self, item: FindBoundingBoxItem) -> str:
        return self.get_output_file_or_dir_for_item(
            item,
            self.output_annotated_images_dir_suffix
        )

    def get_output_file_for_item(self, item: FindBoundingBoxItem) -> str:
        return self.get_output_json_file_for_item(item)


def run(args: argparse.Namespace):
    FindBoundingBoxPipelineFactory(args).run(
        args
    )


def main(argv: Optional[List[str]] = None):
    LOGGER.debug('argv: %r', argv)
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

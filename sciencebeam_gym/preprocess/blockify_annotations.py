import logging
from collections import deque, namedtuple
from abc import ABCMeta, abstractmethod
import math

from six import with_metaclass

from lxml import etree
from pyqtree import Index as PqtreeIndex
from PIL import Image, ImageDraw, ImageColor

from sciencebeam_gym.structured_document.svg import (
    SVG_NSMAP,
    SVG_DOC,
    SVG_RECT,
)

DEFAULT_NEARBY_TOLERANCE = 5


def get_logger():
    return logging.getLogger(__name__)


class AnnotationBlock(object):
    def __init__(self, tag, bounding_box):
        self.tag = tag
        self.bounding_box = bounding_box

    def merge_with(self, other):
        return AnnotationBlock(
            self.tag,
            self.bounding_box.include(other.bounding_box)
        )

    def __str__(self):
        return 'AnnotationBlock({}, {})'.format(self.tag, self.bounding_box)

    def __repr__(self):
        return str(self)


class BlockPoint(object):
    def __init__(self, block, x, y):
        self.block = block
        self.point = (x, y)

    def __str__(self):
        return 'BlockPoint({}, {})'.format(self.block, self.point)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.point)

    def __getitem__(self, index):
        return self.point[index]


def _to_bbox(bb):
    return (bb.x, bb.y, bb.x + bb.width - 1, bb.y + bb.height - 1)


ProcessedWrapper = namedtuple('ProcessedWrapper', field_names=['data', 'deleted'])


class DeletableWrapper(object):
    def __init__(self, data):
        self.data = data
        self.deleted = False

    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        return self.data == other.data


class BlockSearch(object):
    def __init__(self, blocks):
        bboxs = [block.bounding_box for block in blocks]
        xmax = max([bb.x + bb.width for bb in bboxs])
        ymax = max([bb.y + bb.height for bb in bboxs])
        self.spindex = PqtreeIndex(bbox=(0, 0, xmax, ymax))
        self.wrapper_map = {}
        for block in blocks:
            wrapper = DeletableWrapper(block)
            self.wrapper_map[block] = wrapper
            self.spindex.insert(wrapper, _to_bbox(block.bounding_box))

    def find_intersection_with(self, search_bounding_box):
        return [
            wrapper.data
            for wrapper in self.spindex.intersect(_to_bbox(search_bounding_box))
            if not wrapper.deleted
        ]

    def remove(self, block):
        wrapper = self.wrapper_map.get(block)
        if wrapper is not None:
            wrapper.deleted = True


def merge_blocks(blocks, nearby_tolerance=0):
    if len(blocks) <= 1:
        return blocks
    merged_blocks = deque()
    logger = get_logger()
    logger.debug('nearby_tolerance: %s', nearby_tolerance)
    logger.debug('blocks: %s', blocks)
    logger.debug('bboxs: %s', [_to_bbox(block.bounding_box) for block in blocks])
    tags = sorted({b.tag for b in blocks})
    logger.debug('tags: %s', tags)
    remaining_blocks = deque(blocks)
    search_by_tag = {
        tag: BlockSearch([b for b in remaining_blocks if b.tag == tag])
        for tag in tags
    }
    while len(remaining_blocks) >= 2:
        merged_block = remaining_blocks.popleft()
        search = search_by_tag[merged_block.tag]
        search.remove(merged_block)
        search_bounding_box = merged_block.bounding_box.with_margin(1 + nearby_tolerance, 0)
        logger.debug('search_bounding_box: %s (%s)',
                     search_bounding_box, _to_bbox(search_bounding_box))
        neighbours = search.find_intersection_with(search_bounding_box)
        logger.debug('neighbours: %s', neighbours)
        neighbours_blocks_count = 0
        for neighbour in neighbours:
            if neighbour.tag == merged_block.tag:
                merged_block = merged_block.merge_with(neighbour)
                search.remove(neighbour)
                remaining_blocks.remove(neighbour)
                neighbours_blocks_count += 1
        if neighbours_blocks_count == 0 or len(remaining_blocks) == 0:
            logger.debug(
                'no or all remaining blocks merged, mark block as merged: %d',
                neighbours_blocks_count
            )
            merged_blocks.append(merged_block)
        else:
            logger.debug(
                'some but not all remaining blocks merged, continue search: %d',
                neighbours_blocks_count
            )
            remaining_blocks.appendleft(merged_block)
    result = list(merged_blocks) + list(remaining_blocks)
    return result


def expand_bounding_box(bb):
    return bb.with_margin(4, 2)


def expand_block(block):
    return AnnotationBlock(block.tag, expand_bounding_box(block.bounding_box))


def expand_blocks(blocks):
    return [expand_block(block) for block in blocks]


def annotation_document_page_to_annotation_blocks(structured_document, page):
    tags_and_tokens = (
        (structured_document.get_tag_value(token), token)
        for line in structured_document.get_lines_of_page(page)
        for token in structured_document.get_tokens_of_line(line)
    )
    tags_and_bounding_boxes = (
        (tag, structured_document.get_bounding_box(token))
        for tag, token in tags_and_tokens
        if tag
    )
    return [
        AnnotationBlock(tag, bounding_box)
        for tag, bounding_box in tags_and_bounding_boxes
        if bounding_box
    ]


def annotation_document_page_to_merged_blocks(structured_document, page, **kwargs):
    return merge_blocks(
        annotation_document_page_to_annotation_blocks(structured_document, page),
        **kwargs
    )


def extend_color_map_for_tags(color_map, tags):
    updated_color_map = dict(color_map)
    for tag in tags:
        if tag not in updated_color_map:
            updated_color_map[tag] = (
                max(updated_color_map.values()) + 1 if len(updated_color_map) > 0 else 1
            )
    return updated_color_map


def extend_color_map_for_blocks(color_map, blocks):
    return extend_color_map_for_tags(
        color_map,
        sorted({b.tag for b in blocks})
    )


class AbstractSurface(object, with_metaclass(ABCMeta)):
    @abstractmethod
    def rect(self, bounding_box, color, tag=None):
        pass


class SvgSurface(AbstractSurface):
    def __init__(self, width, height, background):
        if not (width and height):
            raise AttributeError('width and height are required')

        self.svg_root = etree.Element(SVG_DOC, nsmap=SVG_NSMAP, attrib={
            'width': str(width),
            'height': str(height)
        })

        if background:
            self.svg_root.append(etree.Element(SVG_RECT, attrib={
                'width': '100%',
                'height': '100%',
                'fill': background,
                'class': 'background'
            }))

    def rect(self, bounding_box, color, tag=None):
        attrib = {
            'class': str(tag),
            'shape-rendering': 'crispEdges',
            'x': str(bounding_box.x),
            'y': str(bounding_box.y),
            'width': str(bounding_box.width),
            'height': str(bounding_box.height)
        }
        if color:
            attrib['fill'] = str(color)
        rect = etree.Element(SVG_RECT, attrib=attrib)
        self.svg_root.append(rect)
        return rect


def color_to_tuple(color):
    if isinstance(color, tuple):
        return color
    return ImageColor.getrgb(color)


class ImageSurface(AbstractSurface):
    def __init__(self, width, height, background):
        if not (width and height):
            raise AttributeError('width and height are required')

        width = int(math.ceil(width))
        height = int(math.ceil(height))
        if background:
            self.image = Image.new('RGB', (width, height), color_to_tuple(background))
        else:
            self.image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        self._draw = ImageDraw.Draw(self.image)

    def rect(self, bounding_box, color, tag=None):
        if color is None:
            return
        self._draw.rectangle(
            (
                (bounding_box.x, bounding_box.y),
                (bounding_box.x + bounding_box.width, bounding_box.y + bounding_box.height)
            ),
            fill=color_to_tuple(color)
        )


def annotated_blocks_to_surface(blocks, surface, color_map):
    for block in blocks:
        color = color_map.get(block.tag)
        surface.rect(block.bounding_box, color, block.tag)


def annotated_blocks_to_svg(blocks, color_map, width=None, height=None, background=None):
    surface = SvgSurface(width, height, background)
    annotated_blocks_to_surface(blocks, surface, color_map)
    return surface.svg_root


def annotated_blocks_to_image(
        blocks, color_map, width=None, height=None, background=None,
        scale_to_size=None):

    surface = ImageSurface(width, height, background)
    annotated_blocks_to_surface(blocks, surface, color_map)
    image = surface.image
    if scale_to_size:
        image = image.resize(scale_to_size, Image.NEAREST)
    return image

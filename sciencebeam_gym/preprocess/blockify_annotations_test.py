import logging

from sciencebeam_gym.utils.bounding_box import (
    BoundingBox
)

from sciencebeam_gym.structured_document import (
    SimpleStructuredDocument,
    SimpleLine,
    SimpleToken,
    B_TAG_PREFIX
)

from sciencebeam_gym.structured_document.svg import (
    SVG_NS
)

from sciencebeam_gym.preprocess.blockify_annotations import (
    annotation_document_page_to_annotation_blocks,
    annotation_document_page_to_merged_blocks,
    annotated_blocks_to_svg,
    annotated_blocks_to_image,
    merge_blocks,
    AnnotationBlock
)

TAG1 = 'tag1'
TAG2 = 'tag2'

DEFAULT_SVGX_WIDTH = 10
DEFAULT_FONT_SIZE = 10
DEFAULT_COLOR = 'red'

DEFAULT_BOUNDING_BOX = BoundingBox(0, 0, 16, 10)

DEFAULT_NEARBY_TOLERANCE = 10


def setup_module():
    logging.basicConfig(level=logging.DEBUG)


class TestAnnotatedBlocksToSvg(object):
    def test_should_create_rect_for_single_annotated_block(self):
        blocks = [
            AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)
        ]

        result_svg = annotated_blocks_to_svg(blocks, color_map={
            TAG1: DEFAULT_COLOR
        }, width=100, height=100)
        result_rect_elements = result_svg.xpath('svg:rect', namespaces={'svg': SVG_NS})
        assert len(result_rect_elements) == 1
        assert result_rect_elements[0].attrib['class'] == TAG1
        assert result_rect_elements[0].attrib['fill'] == DEFAULT_COLOR

    def test_should_add_background(self):
        blocks = [
            AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)
        ]

        result_svg = annotated_blocks_to_svg(blocks, color_map={
            TAG1: '#123'
        }, width=100, height=100, background=DEFAULT_COLOR)
        background_elements = result_svg.xpath(
            'svg:rect[@class="background"]', namespaces={'svg': SVG_NS}
        )
        assert len(background_elements) == 1
        assert background_elements[0].attrib['fill'] == DEFAULT_COLOR


class TestAnnotatedBlocksToImage(object):
    def test_should_create_rect_for_single_annotated_block(self):
        blocks = [
            AnnotationBlock(TAG1, BoundingBox(0, 0, 1, 1))
        ]

        image = annotated_blocks_to_image(blocks, color_map={
            TAG1: (0, 255, 0)
        }, width=3, height=3)
        assert image.getpixel((0, 0)) == (0, 255, 0, 255)

    def test_should_accept_float_image_size(self):
        blocks = [
            AnnotationBlock(TAG1, BoundingBox(0, 0, 1, 1))
        ]

        image = annotated_blocks_to_image(blocks, color_map={
            TAG1: (0, 255, 0)
        }, width=3.1, height=3.9)
        assert image.size == (4, 4)

    def test_should_convert_rect_color_name(self):
        blocks = [
            AnnotationBlock(TAG1, BoundingBox(0, 0, 1, 1))
        ]

        image = annotated_blocks_to_image(blocks, color_map={
            TAG1: 'green'
        }, width=3, height=3)
        assert image.getpixel((0, 0)) == (0, 128, 0, 255)

    def test_should_ignore_unmapped_tag(self):
        blocks = [
            AnnotationBlock(TAG1, BoundingBox(0, 0, 1, 1))
        ]

        image = annotated_blocks_to_image(blocks, color_map={
        }, width=3, height=3)
        assert image.getpixel((0, 0)) == (255, 255, 255, 0)

    def test_should_add_background(self):
        width = 3
        height = 2
        image = annotated_blocks_to_image([], color_map={
            TAG1: '#123'
        }, width=width, height=height, background=(255, 0, 0))
        data = list(image.getdata())
        assert data == [(255, 0, 0)] * (width * height)

    def test_should_convert_background_color_name(self):
        width = 3
        height = 2
        image = annotated_blocks_to_image([], color_map={
            TAG1: '#123'
        }, width=width, height=height, background='red')
        data = list(image.getdata())
        assert data == [(255, 0, 0)] * (width * height)


class TestAnnotationDocumentPageToAnnotationBlocks(object):
    def test_should_convert_single_token_to_block_with_same_bounding_box(self):
        token = SimpleToken('test', tag=TAG1, bounding_box=DEFAULT_BOUNDING_BOX)
        structured_document = SimpleStructuredDocument(lines=[SimpleLine([token])])

        blocks = annotation_document_page_to_annotation_blocks(
            structured_document,
            structured_document.get_pages()[0]
        )
        assert len(blocks) == 1

        block = blocks[0]
        assert block.tag == TAG1
        assert block.bounding_box == DEFAULT_BOUNDING_BOX

    def test_should_strip_tag_prefix(self):
        token = SimpleToken(
            'test', tag=TAG1, tag_prefix=B_TAG_PREFIX,
            bounding_box=DEFAULT_BOUNDING_BOX
        )
        assert token.get_tag() == B_TAG_PREFIX + TAG1
        structured_document = SimpleStructuredDocument(lines=[SimpleLine([token])])

        blocks = annotation_document_page_to_annotation_blocks(
            structured_document,
            structured_document.get_pages()[0]
        )
        assert [b.tag for b in blocks] == [TAG1]

    def test_should_ignore_block_without_bounding_box(self):
        token = SimpleToken('test')
        structured_document = SimpleStructuredDocument(lines=[SimpleLine([token])])
        structured_document.set_tag(token, TAG1)

        blocks = annotation_document_page_to_annotation_blocks(
            structured_document,
            structured_document.get_pages()[0]
        )
        assert len(blocks) == 0


class TestAnnotationDocumentPageToMergedBlocks(object):
    def test_should_convert_single_token_to_block_with_same_bounding_box(self):
        token = SimpleToken('test', tag=TAG1, bounding_box=DEFAULT_BOUNDING_BOX)
        structured_document = SimpleStructuredDocument(lines=[SimpleLine([token])])

        blocks = annotation_document_page_to_merged_blocks(
            structured_document,
            structured_document.get_pages()[0]
        )
        assert len(blocks) == 1

        block = blocks[0]
        assert block.tag == TAG1
        assert block.bounding_box == DEFAULT_BOUNDING_BOX


class TestMergeBlocks(object):
    def test_should_return_same_single_blocks(self):
        block = AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)

        merged_blocks = merge_blocks([block])
        assert merged_blocks == [block]

    def test_should_merge_right_adjacent_block_with_same_tag(self):
        block1 = AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)
        block2 = AnnotationBlock(
            TAG1,
            block1.bounding_box.move_by(block1.bounding_box.width, 0)
        )

        merged_blocks = merge_blocks([block1, block2])
        assert [b.tag for b in merged_blocks] == [TAG1]
        assert merged_blocks[0].bounding_box == block1.bounding_box.include(block2.bounding_box)

    def test_should_not_merge_right_adjacent_block_with_same_different_tag(self):
        block1 = AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)
        block2 = AnnotationBlock(
            TAG2,
            block1.bounding_box.move_by(block1.bounding_box.width, 0)
        )

        merged_blocks = merge_blocks([block1, block2])
        assert [b.tag for b in merged_blocks] == [TAG1, TAG2]

    def test_should_merge_multiple_separate_right_adjacent_blocks(self):
        block1 = AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)
        block2 = AnnotationBlock(
            TAG1,
            block1.bounding_box.move_by(block1.bounding_box.width, 0)
        )

        block3 = AnnotationBlock(
            TAG2,
            block1.bounding_box.move_by(block1.bounding_box.width * 2, 0)
        )
        block4 = AnnotationBlock(
            TAG2,
            block3.bounding_box.move_by(block3.bounding_box.width, 0)
        )

        merged_blocks = merge_blocks([block1, block2, block3, block4])
        assert [b.tag for b in merged_blocks] == [TAG1, TAG2]
        assert merged_blocks[0].bounding_box == block1.bounding_box.include(block2.bounding_box)
        assert merged_blocks[1].bounding_box == block3.bounding_box.include(block4.bounding_box)

    def test_should_merge_multiple_sequential_right_adjacent_blocks(self):
        block1 = AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)
        block2 = AnnotationBlock(
            TAG1,
            block1.bounding_box.move_by(block1.bounding_box.width, 0)
        )
        block3 = AnnotationBlock(
            TAG1,
            block2.bounding_box.move_by(block2.bounding_box.width, 0)
        )

        merged_blocks = merge_blocks([block1, block2, block3])
        assert [b.tag for b in merged_blocks] == [TAG1]

        assert merged_blocks[0].bounding_box == (
            block1.bounding_box.include(block2.bounding_box).include(block3.bounding_box)
        )

    def test_should_merge_right_nearby_block_with_same_tag(self):
        block1 = AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)
        block2 = AnnotationBlock(
            TAG1,
            block1.bounding_box.move_by(block1.bounding_box.width + DEFAULT_NEARBY_TOLERANCE, 0)
        )

        merged_blocks = merge_blocks([block1, block2], nearby_tolerance=DEFAULT_NEARBY_TOLERANCE)
        assert [b.tag for b in merged_blocks] == [TAG1]
        assert merged_blocks[0].bounding_box == block1.bounding_box.include(block2.bounding_box)

    def test_should_not_merge_too_far_away_block_with_same_tag(self):
        block1 = AnnotationBlock(TAG1, DEFAULT_BOUNDING_BOX)
        block2 = AnnotationBlock(
            TAG1,
            block1.bounding_box.move_by(block1.bounding_box.width + DEFAULT_NEARBY_TOLERANCE + 1, 0)
        )

        merged_blocks = merge_blocks([block1, block2], nearby_tolerance=DEFAULT_NEARBY_TOLERANCE)
        assert [b.tag for b in merged_blocks] == [TAG1, TAG1]

    def test_should_merge_right_nearby_block_with_same_tag_using_fractions(self):
        block1 = AnnotationBlock(TAG1, BoundingBox(10.5, 10.5, 10.9, 26.3))
        block2 = AnnotationBlock(
            TAG1,
            block1.bounding_box.move_by(block1.bounding_box.width + DEFAULT_NEARBY_TOLERANCE, 0)
        )

        merged_blocks = merge_blocks([block1, block2], nearby_tolerance=DEFAULT_NEARBY_TOLERANCE)
        assert [b.tag for b in merged_blocks] == [TAG1]
        assert merged_blocks[0].bounding_box == block1.bounding_box.include(block2.bounding_box)

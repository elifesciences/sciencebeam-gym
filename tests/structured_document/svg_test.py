from __future__ import absolute_import

from lxml.builder import ElementMaker

from sciencebeam_gym.utils.bounding_box import (
    BoundingBox
)

from sciencebeam_gym.structured_document.svg import (
    SvgStructuredDocument,
    SvgStyleClasses,
    SVG_NS,
    SVG_VIEWBOX_ATTRIB,
    SVGE_BOUNDING_BOX,
    format_bounding_box
)


E = ElementMaker(namespace=SVG_NS)
SVG_TEXT_BLOCK = lambda *args: E.g({'class': 'block'}, *args)
SVG_TEXT_LINE = lambda *args: E.g({'class': 'line'}, *args)
SVG_TEXT = lambda *args: E.text(*args)

TAG_1 = 'tag1'
TAG_2 = 'tag2'

SCOPE_1 = 'scope1'


class TestSvgStructuredDocument(object):
    def test_should_return_root_as_pages(self):
        root = E.svg()
        doc = SvgStructuredDocument(root)
        assert list(doc.get_pages()) == [root]

    def test_should_find_lines_of_page_without_blocks(self):
        lines = [
            SVG_TEXT_LINE(),
            SVG_TEXT_LINE()
        ]
        doc = SvgStructuredDocument(
            E.svg(
                *lines
            )
        )
        page = doc.get_pages()[0]
        assert list(doc.get_lines_of_page(page)) == lines

    def test_should_find_lines_of_page_with_blocks(self):
        lines = [
            SVG_TEXT_LINE(),
            SVG_TEXT_LINE()
        ]
        doc = SvgStructuredDocument(
            E.svg(
                SVG_TEXT_BLOCK(
                    *lines
                )
            )
        )
        page = doc.get_pages()[0]
        assert list(doc.get_lines_of_page(page)) == lines

    def test_should_find_tokens_of_line(self):
        tokens = [
            SVG_TEXT(),
            SVG_TEXT()
        ]
        line = SVG_TEXT_LINE(*tokens)
        doc = SvgStructuredDocument(
            E.svg(
                line,
                SVG_TEXT_LINE(SVG_TEXT())
            )
        )
        assert list(doc.get_tokens_of_line(line)) == tokens

    def test_should_tag_text_as_line_no(self):
        text = SVG_TEXT()
        doc = SvgStructuredDocument(
            E.svg(
                SVG_TEXT_LINE(text)
            )
        )
        doc.set_tag(text, SvgStyleClasses.LINE_NO)
        assert text.attrib['class'] == SvgStyleClasses.LINE_NO

    def test_should_calculate_default_bounding_box(self):
        text = SVG_TEXT('a', {
            'x': '10',
            'y': '11',
            'font-size': '100'
        })
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(text)))
        assert doc.get_bounding_box(text) == BoundingBox(10, 11, 100 * 0.8, 100)

    def test_should_estimate_width_based_on_number_of_characters(self):
        s = 'abc'
        text = SVG_TEXT(s, {
            'x': '10',
            'y': '11',
            'font-size': '100'
        })
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(text)))
        assert doc.get_bounding_box(text) == BoundingBox(
            10, 11, 100 * 0.8 * len(s), 100
        )

    def test_should_not_return_bounding_box_if_font_size_is_missing(self):
        text = SVG_TEXT({
            'x': '10',
            'y': '11'
        })
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(text)))
        assert doc.get_bounding_box(text) is None

    def test_should_use_bounding_box_if_available(self):
        bounding_box = BoundingBox(11, 12, 101, 102)
        text = SVG_TEXT('a', {
            'x': '10',
            'y': '11',
            'font-size': '100',
            SVGE_BOUNDING_BOX: format_bounding_box(bounding_box)
        })
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(text)))
        assert doc.get_bounding_box(text) == bounding_box

    def test_should_be_able_to_set_bounding_box(self):
        bounding_box = BoundingBox(11, 12, 101, 102)
        text = SVG_TEXT('a', {
            'x': '10',
            'y': '11',
            'font-size': '100'
        })
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(text)))
        doc.set_bounding_box(text, bounding_box)
        assert text.attrib[SVGE_BOUNDING_BOX] == format_bounding_box(bounding_box)

    def test_should_use_viewbox_if_available(self):
        bounding_box = BoundingBox(11, 12, 101, 102)
        page = E.svg({
            SVG_VIEWBOX_ATTRIB: format_bounding_box(bounding_box)
        })
        doc = SvgStructuredDocument(page)
        assert doc.get_bounding_box(page) == bounding_box

    def test_should_set_tag_without_scope(self):
        token = SVG_TEXT('test')
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(token)))
        doc.set_tag(token, TAG_1)
        assert doc.get_tag(token) == TAG_1

    def test_should_set_tag_with_scope(self):
        token = SVG_TEXT('test')
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(token)))
        doc.set_tag(token, TAG_1, scope=SCOPE_1)
        assert doc.get_tag(token, scope=SCOPE_1) == TAG_1
        assert doc.get_tag(token) is None

    def test_should_set_tag_with_level(self):
        token = SVG_TEXT('test')
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(token)))
        doc.set_tag(token, TAG_1, level=2)
        assert doc.get_tag(token, level=2) == TAG_1
        assert doc.get_tag(token) is None

    def test_should_return_all_tag_by_scope(self):
        token = SVG_TEXT('test')
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(token)))
        doc.set_tag(token, TAG_1)
        doc.set_tag(token, TAG_2, scope=SCOPE_1)
        assert doc.get_tag(token) == TAG_1
        assert doc.get_tag(token, scope=SCOPE_1) == TAG_2
        assert doc.get_tag_by_scope(token) == {None: TAG_1, SCOPE_1: TAG_2}

    def test_should_clear_tag_when_setting_tag_to_none(self):
        token = SVG_TEXT('test')
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(token)))
        doc.set_tag(token, TAG_1)
        doc.set_tag(token, TAG_1, scope=SCOPE_1)
        doc.set_tag(token, None)
        doc.set_tag(token, None, scope=SCOPE_1)
        assert doc.get_tag(token) is None
        assert doc.get_tag(token, scope=SCOPE_1) is None

    def test_should_not_fail_setting_empty_tag_to_none(self):
        token = SVG_TEXT('test')
        doc = SvgStructuredDocument(E.svg(SVG_TEXT_LINE(token)))
        doc.set_tag(token, None)
        doc.set_tag(token, None, scope=SCOPE_1)
        assert doc.get_tag(token) is None
        assert doc.get_tag(token, scope=SCOPE_1) is None

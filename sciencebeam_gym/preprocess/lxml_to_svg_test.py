from lxml.builder import E

from sciencebeam_gym.utils.bounding_box import (
  BoundingBox
)

from sciencebeam_gym.structured_document.svg import (
  SVG_TEXT,
  SVG_G,
  SVG_DOC,
  SVG_NS,
  SVGE_BOUNDING_BOX,
  parse_bounding_box
)

from sciencebeam_gym.preprocess.lxml_to_svg import (
  iter_svg_pages_for_lxml
)

SOME_TEXT = "some text"
SOME_X = "10"
SOME_Y = "20"
SOME_BASE = "25"
SOME_WIDTH = "31"
SOME_HEIGHT = "30"
SOME_FONT_SIZE = "40"
SOME_FONT_FAMILY = "Fontastic"
SOME_FONT_COLOR = '#123'

class LXML(object):
  X = 'x'
  Y = 'y'
  BASE = 'base'
  WIDTH = 'width'
  HEIGHT = 'height'
  FONT_SIZE = 'font-size'
  FONT_NAME = 'font-name'
  FONT_COLOR = 'font-color'

class SVG(object):
  X = 'x'
  Y = 'y'
  HEIGHT = 'height'
  FONT_SIZE = 'font-size'
  FONT_FAMILY = 'font-family'
  FILL = 'fill'
  BOUNDING_BOX = SVGE_BOUNDING_BOX

COMMON_LXML_TOKEN_ATTRIBS = {
  LXML.X: SOME_X,
  LXML.Y: SOME_Y,
  LXML.WIDTH: SOME_WIDTH,
  LXML.HEIGHT: SOME_HEIGHT,
  LXML.FONT_SIZE: SOME_FONT_SIZE,
  LXML.FONT_NAME: SOME_FONT_FAMILY,
  LXML.FONT_COLOR: SOME_FONT_COLOR
}

def dict_extend(*dicts):
  d = dict()
  for x in dicts:
    d.update(x)
  return d

class TestIterSvgPagesForLxml(object):
  def test_should_return_one_page(self):
    lxml_root = E.DOCUMENT(
      E.PAGE(
      )
    )
    svg_pages = list(iter_svg_pages_for_lxml(lxml_root))
    assert len(svg_pages) == 1

  def test_should_return_multiple_pages(self):
    lxml_root = E.DOCUMENT(
      E.PAGE(
      ),
      E.PAGE(
      ),
      E.PAGE(
      )
    )
    svg_pages = list(iter_svg_pages_for_lxml(lxml_root))
    assert len(svg_pages) == 3

  def test_should_set_svg_dimensions(self):
    lxml_root = E.DOCUMENT(
      E.PAGE(
        width='600',
        height='800'
      )
    )
    svg_pages = list(iter_svg_pages_for_lxml(lxml_root))
    assert len(svg_pages) == 1
    assert svg_pages[0].attrib.get('viewBox') == '0 0 600 800'

  def test_should_add_background_rect(self):
    lxml_root = E.DOCUMENT(
      E.PAGE(
        width='600',
        height='800'
      )
    )
    svg_pages = list(iter_svg_pages_for_lxml(lxml_root))
    assert len(svg_pages) == 1
    background_rect = svg_pages[0].xpath(
      'svg:rect[@class="background"]',
      namespaces={'svg': SVG_NS}
    )
    assert len(background_rect) == 1

  def test_should_create_text_node_with_common_attributes(self):
    lxml_root = E.DOCUMENT(
      E.PAGE(
        E.TEXT(
          E.TOKEN(
            SOME_TEXT,
            COMMON_LXML_TOKEN_ATTRIBS
          )
        )
      )
    )
    svg_pages = list(iter_svg_pages_for_lxml(lxml_root))
    assert len(svg_pages) == 1
    first_page = svg_pages[0]
    svg_text = first_page.find('.//' + SVG_TEXT)
    assert svg_text is not None
    assert svg_text.text == SOME_TEXT
    assert float(svg_text.attrib[SVG.X]) == float(SOME_X)
    assert float(svg_text.attrib[SVG.Y]) == float(SOME_Y)
    assert float(svg_text.attrib[SVG.FONT_SIZE]) == float(SOME_FONT_SIZE)
    assert svg_text.attrib[SVG.FONT_FAMILY] == SOME_FONT_FAMILY
    assert svg_text.attrib[SVG.FILL] == SOME_FONT_COLOR
    assert parse_bounding_box(svg_text.attrib.get(SVG.BOUNDING_BOX)) == BoundingBox(
      float(COMMON_LXML_TOKEN_ATTRIBS[LXML.X]),
      float(COMMON_LXML_TOKEN_ATTRIBS[LXML.Y]),
      float(COMMON_LXML_TOKEN_ATTRIBS[LXML.WIDTH]),
      float(COMMON_LXML_TOKEN_ATTRIBS[LXML.HEIGHT])
    )

  def test_should_use_base_as_y_in_svg_if_available(self):
    lxml_root = E.DOCUMENT(
      E.PAGE(
        E.TEXT(
          E.TOKEN(
            SOME_TEXT,
            dict_extend(COMMON_LXML_TOKEN_ATTRIBS, {
              LXML.BASE: SOME_BASE
            })
          )
        )
      )
    )
    svg_pages = list(iter_svg_pages_for_lxml(lxml_root))
    assert len(svg_pages) == 1
    first_page = svg_pages[0]
    svg_text = first_page.find('.//' + SVG_TEXT)
    assert float(svg_text.attrib[SVG.Y]) == float(SOME_BASE)

  def test_should_keep_text_block_structure_without_block(self):
    lxml_root = E.DOCUMENT(
      E.PAGE(
        E.TEXT(
          E.TOKEN(
            SOME_TEXT,
            dict_extend(COMMON_LXML_TOKEN_ATTRIBS, {
              LXML.BASE: SOME_BASE
            })
          )
        )
      )
    )
    svg_pages = list(iter_svg_pages_for_lxml(lxml_root))
    assert len(svg_pages) == 1
    first_page = svg_pages[0]
    svg_text = first_page.find('.//' + SVG_TEXT)
    assert svg_text is not None
    assert svg_text.getparent().tag == SVG_G
    assert svg_text.getparent().getparent().tag == SVG_DOC

  def test_should_keep_text_block_structure_with_block(self):
    lxml_root = E.DOCUMENT(
      E.PAGE(
        E.BLOCK(
          E.TEXT(
            E.TOKEN(
              SOME_TEXT,
              dict_extend(COMMON_LXML_TOKEN_ATTRIBS, {
                LXML.BASE: SOME_BASE
              })
            )
          )
        )
      )
    )
    svg_pages = list(iter_svg_pages_for_lxml(lxml_root))
    assert len(svg_pages) == 1
    first_page = svg_pages[0]
    svg_text = first_page.find('.//' + SVG_TEXT)
    assert svg_text is not None
    assert svg_text.getparent().tag == SVG_G
    assert svg_text.getparent().getparent().tag == SVG_G
    assert svg_text.getparent().getparent().getparent().tag == SVG_DOC

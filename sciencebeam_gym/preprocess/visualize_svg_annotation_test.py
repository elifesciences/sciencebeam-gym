import logging

from lxml import etree

from sciencebeam_gym.structured_document.svg import (
  SVG_NSMAP,
  SVG_DOC,
  SVG_TEXT,
  SVG_TAG_ATTRIB
)

from sciencebeam_gym.preprocess.visualize_svg_annotation import (
  visualize_svg_annotations,
  style_block_for_tags,
  style_block_for_tag,
  style_props_for_tags,
  render_style_props,
  color_for_tag
)

TAG1 = 'tag1'
TAG2 = 'tag2'


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('test')


def _create_xml_node(tag, text=None, attrib=None):
  node = etree.Element(tag)
  if text is not None:
    node.text = text
  if attrib is not None:
    for k, v in attrib.items():
      node.attrib[k] = str(v)
  return node

def _create_tagged_text(svga_tags):
  return _create_xml_node(SVG_TEXT, attrib={
    SVG_TAG_ATTRIB: svga_tags
  })


def test_add_style_block_for_single_tag_on_multiple_nodes():
  svg_root = etree.Element(SVG_DOC, nsmap=SVG_NSMAP)

  svg_root.append(_create_tagged_text(TAG1))
  svg_root.append(_create_tagged_text(TAG1))

  result_svg = visualize_svg_annotations(svg_root)
  style_block = result_svg.find('style')

  assert style_block is not None
  assert style_block.text == style_block_for_tags([TAG1])

def test_add_style_block_for_multiple_tags_on_separate_nodes():
  svg_root = etree.Element(SVG_DOC, nsmap=SVG_NSMAP)

  svg_root.append(_create_tagged_text(TAG1))
  svg_root.append(_create_tagged_text(TAG2))

  result_svg = visualize_svg_annotations(svg_root)
  style_block = result_svg.find('style')

  assert style_block is not None
  assert style_block.text == style_block_for_tags([TAG1, TAG2])

def test_add_style_block_for_multiple_tags_on_same_node():
  svg_root = etree.Element(SVG_DOC, nsmap=SVG_NSMAP)

  svg_root.append(_create_tagged_text(' '.join([TAG1, TAG2])))

  result_svg = visualize_svg_annotations(svg_root)
  style_block = result_svg.find('style')

  assert style_block is not None
  assert style_block.text == style_block_for_tags([TAG1, TAG2])

def test_add_title_with_tags():
  svg_root = etree.Element(SVG_DOC, nsmap=SVG_NSMAP)

  svg_root.append(_create_tagged_text(TAG1))

  result_svg = visualize_svg_annotations(svg_root)
  text_node = result_svg.find(SVG_TEXT + '/title')

  assert text_node is not None
  assert text_node.text == TAG1

def test_style_block_for_single_tag():
  style_block_text = style_block_for_tags([TAG1])

  assert style_block_text == (
    style_block_for_tag(TAG1, style_props_for_tags([TAG1])[TAG1])
  )

def test_style_block_for_multiple_tags():
  style_block_text = style_block_for_tags([TAG1, TAG2])
  style_props_map = style_props_for_tags([TAG1, TAG2])

  assert style_block_text == (
    '\n\n'.join([
      style_block_for_tag(TAG1, style_props_map[TAG1]),
      style_block_for_tag(TAG2, style_props_map[TAG2])
    ])
  )

def test_style_block_for_tag():
  style_props = style_props_for_tags([TAG1])[TAG1]
  style_block_text = style_block_for_tag(TAG1, style_props)

  assert (
    style_block_text ==
    'text[class~="' + TAG1 + '"] {\n' + render_style_props(style_props) + '\n}'
  )

def test_color_for_tag_should_be_different_for_different_tags():
  assert color_for_tag(TAG1) != color_for_tag(TAG2)

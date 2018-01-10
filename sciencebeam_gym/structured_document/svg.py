from sciencebeam_gym.utils.bounding_box import (
  BoundingBox
)

from sciencebeam_gym.utils.xml import (
  set_or_remove_attrib
)

from sciencebeam_gym.structured_document import (
  AbstractStructuredDocument,
  get_scoped_attrib_name,
  get_attrib_by_scope
)

SVG_NS = 'http://www.w3.org/2000/svg'
SVG_NS_PREFIX = '{' + SVG_NS + '}'
SVG_DOC = SVG_NS_PREFIX + 'svg'
SVG_TEXT = SVG_NS_PREFIX + 'text'
SVG_G = SVG_NS_PREFIX + 'g'
SVG_RECT = SVG_NS_PREFIX + 'rect'

SVG_VIEWBOX_ATTRIB = 'viewBox'

SVG_TAG_ATTRIB = 'class'

SVGE_NS = 'http://www.elifesciences.org/schema/svge'
SVGE_NS_PREFIX = '{' + SVGE_NS + '}'
SVGE_BOUNDING_BOX = SVGE_NS_PREFIX + 'bounding-box'

SCOPED_TAG_ATTRIB_SUFFIX = 'tag'

SVG_NSMAP = {
  None : SVG_NS,
  'svge': SVGE_NS
}

class SvgStyleClasses(object):
  LINE = 'line'
  BLOCK = 'block'
  LINE_NO = 'line_no'

def format_bounding_box(bounding_box):
  return '%s %s %s %s' % (bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height)

def parse_bounding_box(bounding_box_str):
  if not bounding_box_str:
    return None
  x, y, width, height = bounding_box_str.split()
  return BoundingBox(float(x), float(y), float(width), float(height))

def get_node_bounding_box(t):
  attrib = t.attrib
  if SVGE_BOUNDING_BOX in attrib:
    return parse_bounding_box(attrib[SVGE_BOUNDING_BOX])
  if SVG_VIEWBOX_ATTRIB in attrib:
    return parse_bounding_box(attrib[SVG_VIEWBOX_ATTRIB])
  if not ('font-size' in attrib and 'x' in attrib and 'y' in attrib):
    return None
  font_size = float(attrib['font-size'])
  width = font_size * 0.8 * max(1, len(t.text))
  return BoundingBox(
    float(attrib['x']),
    float(attrib['y']),
    width,
    font_size
  )

def _get_tag_attrib_name(scope):
  return (
    SVGE_NS_PREFIX + get_scoped_attrib_name(SCOPED_TAG_ATTRIB_SUFFIX, scope) if scope
    else SVG_TAG_ATTRIB
  )

class SvgStructuredDocument(AbstractStructuredDocument):
  def __init__(self, root_or_roots):
    if isinstance(root_or_roots, list):
      self.page_roots = root_or_roots
    else:
      self.page_roots = [root_or_roots]

  def get_pages(self):
    return self.page_roots

  def get_lines_of_page(self, page):
    return page.findall('.//{}[@class="{}"]'.format(SVG_G, SvgStyleClasses.LINE))

  def get_tokens_of_line(self, line):
    return line.findall('./{}'.format(SVG_TEXT))

  def get_x(self, parent):
    return parent.attrib.get('x')

  def get_text(self, parent):
    return parent.text

  def get_tag(self, parent, scope=None):
    return parent.attrib.get(_get_tag_attrib_name(scope))

  def set_tag(self, parent, tag, scope=None):
    set_or_remove_attrib(parent.attrib, _get_tag_attrib_name(scope), tag)

  def get_tag_by_scope(self, parent):
    d = {
      k[len(SVGE_NS_PREFIX):]: v
      for k, v in get_attrib_by_scope(parent.attrib, SCOPED_TAG_ATTRIB_SUFFIX).items()
      if k.startswith(SVGE_NS_PREFIX)
    }
    tag = self.get_tag(parent)
    if tag:
      d[None] = tag
    return d

  def get_bounding_box(self, parent):
    return get_node_bounding_box(parent)

  def set_bounding_box(self, parent, bounding_box):
    parent.attrib[SVGE_BOUNDING_BOX] = format_bounding_box(bounding_box)

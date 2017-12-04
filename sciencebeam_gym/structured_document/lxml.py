from sciencebeam_gym.utils.bounding_box import (
  BoundingBox
)

from sciencebeam_gym.structured_document import (
  AbstractStructuredDocument
)

def get_node_bounding_box(t):
  return BoundingBox(
    float(t.attrib['x']),
    float(t.attrib['y']),
    float(t.attrib['width']),
    float(t.attrib['height'])
  )

class LxmlStructuredDocument(AbstractStructuredDocument):
  def __init__(self, root):
    self.root = root

  def get_pages(self):
    return self.root.findall('.//PAGE')

  def get_lines_of_page(self, page):
    return page.findall('.//TEXT')

  def get_tokens_of_line(self, line):
    return line.findall('./TOKEN')

  def get_x(self, parent):
    return parent.attrib.get('x')

  def get_text(self, parent):
    return parent.text

  def get_tag(self, parent):
    return parent.attrib.get('tag')

  def set_tag(self, parent, tag):
    parent.attrib['tag'] = tag

  def get_bounding_box(self, parent):
    return get_node_bounding_box(parent)

  def set_bounding_box(self, parent, bounding_box):
    parent.attrib['x'] = str(bounding_box.x)
    parent.attrib['y'] = str(bounding_box.y)
    parent.attrib['width'] = str(bounding_box.width)
    parent.attrib['height'] = str(bounding_box.height)

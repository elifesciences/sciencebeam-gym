from abc import ABCMeta, abstractmethod

from six import with_metaclass

class AbstractStructuredDocument(object, with_metaclass(ABCMeta)):
  @abstractmethod
  def get_pages(self):
    pass

  @abstractmethod
  def get_lines_of_page(self, page):
    pass

  @abstractmethod
  def get_tokens_of_line(self, line):
    pass

  @abstractmethod
  def get_x(self, parent):
    pass

  @abstractmethod
  def get_text(self, parent):
    pass

  @abstractmethod
  def get_tag(self, parent):
    pass

  @abstractmethod
  def set_tag(self, parent, tag):
    pass

  @abstractmethod
  def get_bounding_box(self, parent):
    pass

  @abstractmethod
  def set_bounding_box(self, parent, bounding_box):
    pass

class SimpleElement(object):
  def __init__(self):
    self._bounding_box = None

  def get_bounding_box(self):
    return self._bounding_box

  def set_bounding_box(self, bounding_box):
    self._bounding_box = bounding_box

class SimpleToken(SimpleElement):
  def __init__(self, text, attrib=None):
    super(SimpleToken, self).__init__()
    self.text = text
    if attrib is None:
      attrib = {}
    self.attrib = attrib
    self._bounding_box = None

  def get_x(self):
    return self.attrib.get('x')

  def get_y(self):
    return self.attrib.get('y')

  def get_tag(self):
    return self.attrib.get('tag')

  def get_text(self):
    return self.text

  def set_tag(self, tag):
    self.attrib['tag'] = tag

class SimpleLine(SimpleElement):
  def __init__(self, tokens):
    super(SimpleLine, self).__init__()
    self.tokens = tokens

class SimpleDocument(SimpleElement):
  def __init__(self, lines):
    super(SimpleDocument, self).__init__()
    self.lines = lines

class SimpleStructuredDocument(AbstractStructuredDocument):
  def __init__(self, root=None, lines=None):
    if lines is not None:
      root = SimpleDocument(lines)
    self.root = root

  def get_pages(self):
    return [self.root]

  def get_lines_of_page(self, page):
    return page.lines

  def get_tokens_of_line(self, line):
    return line.tokens

  def get_x(self, parent):
    return parent.get_x()

  def get_text(self, parent):
    return parent.get_text()

  def get_tag(self, parent):
    return parent.get_tag()

  def set_tag(self, parent, tag):
    return parent.set_tag(tag)

  def get_bounding_box(self, parent):
    return parent.get_bounding_box()

  def set_bounding_box(self, parent, bounding_box):
    parent.set_bounding_box(bounding_box)

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
  def __init__(self, bounding_box=None):
    self._bounding_box = bounding_box

  def get_bounding_box(self):
    return self._bounding_box

  def set_bounding_box(self, bounding_box):
    self._bounding_box = bounding_box

class SimpleToken(SimpleElement):
  def __init__(self, text, attrib=None, tag=None, **kwargs):
    super(SimpleToken, self).__init__(**kwargs)
    self.text = text
    if attrib is None:
      attrib = {}
    self.attrib = attrib
    if tag is not None:
      self.set_tag(tag)

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

class SimplePage(SimpleElement):
  def __init__(self, lines, **kwargs):
    super(SimplePage, self).__init__(**kwargs)
    self.lines = lines

class SimpleStructuredDocument(AbstractStructuredDocument):
  def __init__(self, page_or_pages=None, lines=None):
    if lines is not None:
      pages = [SimplePage(lines)]
    elif page_or_pages is None:
      pages = []
    elif isinstance(page_or_pages, list):
      pages = page_or_pages
    else:
      pages = [page_or_pages]
    self._pages = pages

  def get_pages(self):
    return self._pages

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

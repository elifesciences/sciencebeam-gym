from abc import ABCMeta, abstractmethod
from copy import deepcopy

from six import with_metaclass

def merge_token_tag(
  merged_structured_document, merged_token,
  other_structured_document, other_token,
  source_scope=None, target_scope=None):

  tag = other_structured_document.get_tag(other_token, scope=source_scope)
  merged_structured_document.set_tag(
    merged_token,
    tag,
    scope=target_scope
  )

SCOPE_ATTRIB_SEP = '-'


def get_scoped_attrib_name(name, scope=None):
  return '%s%s%s' % (scope, SCOPE_ATTRIB_SEP, name) if scope else name

def get_attrib_by_scope(attrib, name):
  suffix = '%s%s' % (SCOPE_ATTRIB_SEP, name)
  return {
    (None if k == name else k[:-len(suffix)]): v
    for k, v in attrib.items()
    if k.endswith(suffix) or k == name
  }

SIMPLE_TAG_ATTRIB_NAME = 'tag'

def get_simple_tag_attrib_name(scope):
  return get_scoped_attrib_name(SIMPLE_TAG_ATTRIB_NAME, scope)

class AbstractStructuredDocument(object, with_metaclass(ABCMeta)):
  def clone(self):
    return deepcopy(self)

  def iter_all_tokens(self):
    for page in self.get_pages():
      for line in self.get_lines_of_page(page):
        for token in self.get_tokens_of_line(line):
          yield token

  def merge_with(
    self,
    other_structured_document,
    merge_fn):
    """
    Merges this structured document with another structured document using the merge fn.

    Note: this document will be changed (operate on a clone if that is undesired)
    """

    for merged_token, other_token in zip(
      self.iter_all_tokens(),
      other_structured_document.iter_all_tokens()
      ):
      assert (
        self.get_text(merged_token) ==
        other_structured_document.get_text(other_token)
      )
      merge_fn(
        self, merged_token,
        other_structured_document, other_token
      )

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
  def get_tag(self, parent, scope=None):
    pass

  @abstractmethod
  def set_tag(self, parent, tag, scope=None):
    pass

  @abstractmethod
  def get_tag_by_scope(self, parent):
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
  def __init__(self, text, attrib=None, tag=None, tag_scope=None, **kwargs):
    super(SimpleToken, self).__init__(**kwargs)
    self.text = text
    if attrib is None:
      attrib = {}
    self.attrib = attrib
    if tag is not None:
      self.set_tag(tag, scope=tag_scope)

  def get_x(self):
    return self.attrib.get('x')

  def get_y(self):
    return self.attrib.get('y')

  def get_tag(self, scope=None):
    return self.attrib.get(get_simple_tag_attrib_name(scope))

  def set_tag(self, tag, scope=None):
    self.attrib[get_simple_tag_attrib_name(scope)] = tag

  def get_tag_by_scope(self):
    return get_attrib_by_scope(self.attrib, SIMPLE_TAG_ATTRIB_NAME)

  def get_text(self):
    return self.text

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

  def get_tag(self, parent, scope=None):
    return parent.get_tag(scope=scope)

  def set_tag(self, parent, tag, scope=None):
    return parent.set_tag(tag, scope=scope)

  def get_tag_by_scope(self, parent):
    return parent.get_tag_by_scope()

  def get_bounding_box(self, parent):
    return parent.get_bounding_box()

  def set_bounding_box(self, parent, bounding_box):
    parent.set_bounding_box(bounding_box)

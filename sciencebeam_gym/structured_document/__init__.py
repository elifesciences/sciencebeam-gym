from abc import ABCMeta, abstractmethod
from copy import deepcopy

from six import with_metaclass

B_TAG_PREFIX = 'b-'
I_TAG_PREFIX = 'i-'

SCOPE_ATTRIB_SEP = '-'
LEVEL_ATTRIB_SEP = '_'

SIMPLE_TAG_ATTRIB_NAME = 'tag'

def merge_token_tag(
  merged_structured_document, merged_token,
  other_structured_document, other_token,
  source_scope=None, target_scope=None):

  tag = other_structured_document.get_tag(other_token, scope=source_scope)
  if tag:
    merged_structured_document.set_tag(
      merged_token,
      tag,
      scope=target_scope
    )

def get_scoped_attrib_name(name, scope=None, level=None):
  if level:
    name = 'level%s%s%s' % (level, LEVEL_ATTRIB_SEP, name)
  return '%s%s%s' % (scope, SCOPE_ATTRIB_SEP, name) if scope else name

def get_attrib_by_scope(attrib, name):
  suffix = '%s%s' % (SCOPE_ATTRIB_SEP, name)
  return {
    (None if k == name else k[:-len(suffix)]): v
    for k, v in attrib.items()
    if k.endswith(suffix) or k == name
  }

def get_simple_tag_attrib_name(scope, level=None):
  return get_scoped_attrib_name(SIMPLE_TAG_ATTRIB_NAME, scope, level)

def split_tag_prefix(tag):
  if tag:
    if tag.startswith(B_TAG_PREFIX):
      return B_TAG_PREFIX, tag[len(B_TAG_PREFIX):]
    if tag.startswith(I_TAG_PREFIX):
      return I_TAG_PREFIX, tag[len(I_TAG_PREFIX):]
  return None, tag

def strip_tag_prefix(tag):
  return split_tag_prefix(tag)[1]

def add_tag_prefix(tag, prefix):
  return prefix + tag if prefix and tag else tag

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

  def get_tag_prefix_and_value(self, parent, scope=None):
    return split_tag_prefix(self.get_tag(parent, scope=scope))

  def get_tag_value(self, parent, scope=None):
    return self.get_tag_prefix_and_value(parent, scope=scope)[1]

  def set_tag_with_prefix(self, parent, tag, scope=None, prefix=None):
    self.set_tag(parent, add_tag_prefix(tag, prefix), scope=scope)

  def get_sub_tag(self, parent, scope=None):
    return self.get_tag(parent, scope=scope, level=2)

  def set_sub_tag(self, parent, tag, scope=None):
    self.set_tag(parent, tag, scope=scope, level=2)

  def set_sub_tag_with_prefix(self, parent, tag, scope=None, prefix=None):
    self.set_sub_tag(parent, add_tag_prefix(tag, prefix), scope=scope)

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
  def get_tag(self, parent, scope=None, level=None):
    pass

  @abstractmethod
  def set_tag(self, parent, tag, scope=None, level=None):
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
  def __init__(
    self, text, attrib=None, tag=None, tag_scope=None, tag_prefix=None, **kwargs):
    super(SimpleToken, self).__init__(**kwargs)
    self.text = text
    if attrib is None:
      attrib = {}
    self.attrib = attrib
    if tag is not None:
      self.set_tag(tag, scope=tag_scope, prefix=tag_prefix)

  def get_x(self):
    return self.attrib.get('x')

  def get_y(self):
    return self.attrib.get('y')

  def get_tag(self, scope=None, level=None):
    return self.attrib.get(get_simple_tag_attrib_name(scope=scope, level=level))

  def set_tag(self, tag, scope=None, level=None, prefix=None):
    self.attrib[get_simple_tag_attrib_name(scope=scope, level=level)] = add_tag_prefix(tag, prefix)

  def get_tag_by_scope(self):
    return get_attrib_by_scope(self.attrib, SIMPLE_TAG_ATTRIB_NAME)

  def get_text(self):
    return self.text

  def __repr__(self):
    return '%s(%s)' % (type(self).__name__, self.text)

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

  def get_tag(self, parent, scope=None, level=None):
    return parent.get_tag(scope=scope, level=level)

  def set_tag(self, parent, tag, scope=None, level=None):
    return parent.set_tag(tag, scope=scope, level=level)

  def get_tag_by_scope(self, parent):
    return parent.get_tag_by_scope()

  def get_bounding_box(self, parent):
    return parent.get_bounding_box()

  def set_bounding_box(self, parent, bounding_box):
    parent.set_bounding_box(bounding_box)

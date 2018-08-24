from __future__ import absolute_import

from lxml import etree

from sciencebeam_utils.beam_utils.io import (
  save_file_content
)

from sciencebeam_gym.utils.pages_zip import (
  save_pages
)

from sciencebeam_gym.structured_document.lxml import (
  LxmlStructuredDocument
)

from sciencebeam_gym.structured_document.svg import (
  SvgStructuredDocument
)

def save_lxml_structured_document(filename, lxml_structured_document):
  save_file_content(filename, etree.tostring(lxml_structured_document.root))

def save_svg_structured_document(filename, svg_structured_document):
  return save_pages(filename, '.svg', (
    etree.tostring(svg_page)
    for svg_page in svg_structured_document.get_pages()
  ))

def save_structured_document(filename, structured_document):
  if isinstance(structured_document, LxmlStructuredDocument):
    return save_lxml_structured_document(filename, structured_document)
  if isinstance(structured_document, SvgStructuredDocument):
    return save_svg_structured_document(filename, structured_document)
  raise RuntimeError('unsupported type: %s' % type(structured_document))

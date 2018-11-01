from __future__ import absolute_import

from lxml import etree
from lxml.builder import E

from apache_beam.io.filesystems import FileSystems

from sciencebeam_gym.utils.pages_zip import (
  load_pages
)

from sciencebeam_gym.structured_document.lxml import (
  LxmlStructuredDocument
)

from sciencebeam_gym.structured_document.svg import (
  SvgStructuredDocument
)

class StructuredDocumentType(object):
  LXML = 'lxml'
  SVG_PAGES = 'svg-pages'

def get_structuctured_document_type(filename):
  if filename.endswith('.zip'):
    return StructuredDocumentType.SVG_PAGES
  return StructuredDocumentType.LXML

def load_lxml_structured_document(filename, page_range=None):
  with FileSystems.open(filename) as f:
    structured_document = LxmlStructuredDocument(etree.parse(f).getroot())
    if page_range:
      structured_document = LxmlStructuredDocument(
        E.DOCUMENT(
          *structured_document.get_pages()[
            max(0, page_range[0] - 1):
            page_range[1]
          ]
        )
      )
    return structured_document

def load_svg_pages_structured_document(filename, page_range=None):
  return SvgStructuredDocument([
    etree.parse(svg_f).getroot()
    for svg_f in load_pages(filename, page_range=page_range)
  ])

def load_structured_document(filename, page_range=None):
  structured_document_type = get_structuctured_document_type(filename)
  if structured_document_type == StructuredDocumentType.LXML:
    return load_lxml_structured_document(filename, page_range=page_range)
  if structured_document_type == StructuredDocumentType.SVG_PAGES:
    return load_svg_pages_structured_document(filename, page_range=page_range)
  raise RuntimeError('unsupported structured_document_type: %s (%s)' % (
    structured_document_type, filename
  ))

def load_structured_documents_from_file_list(file_list, page_range=None):
  return (load_structured_document(s, page_range=page_range) for s in file_list)

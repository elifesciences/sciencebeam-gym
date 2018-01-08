from __future__ import absolute_import

from zipfile import ZipFile

from lxml import etree

from apache_beam.io.filesystems import FileSystems

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

def load_lxml_structured_document(filename):
  with FileSystems.open(filename) as f:
    return LxmlStructuredDocument(etree.parse(f))

def load_svg_pages_structured_document(filename):
  with FileSystems.open(filename) as f:
    with ZipFile(f) as zf:
      filenames = zf.namelist()
      svg_roots = []
      for filename in filenames:
        with zf.open(filename) as svg_f:
          svg_roots.append(etree.parse(svg_f))
    return SvgStructuredDocument(svg_roots)

def load_structured_document(filename):
  structured_document_type = get_structuctured_document_type(filename)
  if structured_document_type == StructuredDocumentType.LXML:
    return load_lxml_structured_document(filename)
  if structured_document_type == StructuredDocumentType.SVG_PAGES:
    return load_svg_pages_structured_document(filename)

def load_structured_documents_from_file_list(file_list):
  return (load_structured_document(s) for s in file_list)

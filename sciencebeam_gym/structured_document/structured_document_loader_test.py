from __future__ import absolute_import

from tempfile import NamedTemporaryFile
from zipfile import ZipFile
from mock import patch

from lxml import etree
from lxml.builder import E

import sciencebeam_gym.structured_document.structured_document_loader as structured_document_loader

from sciencebeam_gym.structured_document.structured_document_loader import (
  StructuredDocumentType,
  get_structuctured_document_type,
  load_structured_documents_from_file_list,
  load_lxml_structured_document,
  load_svg_pages_structured_document,
  load_structured_document
)

FILE_1 = 'file1.pdf'
FILE_2 = 'file2.pdf'

class TestLoadLxmlStructuredDocument(object):
  def test_should_load_file(self):
    lxml_content = etree.tostring(E.test('test'))
    with NamedTemporaryFile() as f:
      f.write(lxml_content)
      f.flush()
      structured_document = load_lxml_structured_document(f.name)
      assert etree.tostring(structured_document.root) == lxml_content

class TestLoadSvgPagesStructuredDocument(object):
  def test_should_load_file_with_multiple_pages(self):
    svg_pages_content = [
      etree.tostring(E.svg('page 1')),
      etree.tostring(E.svg('page 2'))
    ]
    with NamedTemporaryFile() as f:
      with ZipFile(f, 'w') as zf:
        for i, svg_page_content in enumerate(svg_pages_content):
          zf.writestr('page-%d.svg' % (1 + i), svg_page_content)
      f.flush()
      structured_document = load_svg_pages_structured_document(f.name)
      assert (
        [etree.tostring(x) for x in structured_document.page_roots] ==
        svg_pages_content
      )

class TestGetStructuredDocumentType(object):
  def test_should_return_lxml_for_lxml_file(self):
    assert get_structuctured_document_type('file.lxml') == StructuredDocumentType.LXML

  def test_should_return_lxml_for_lxml_gz_file(self):
    assert get_structuctured_document_type('file.lxml.gz') == StructuredDocumentType.LXML

  def test_should_return_lxml_for_svg_zip_file(self):
    assert get_structuctured_document_type('file.svg.zip') == StructuredDocumentType.SVG_PAGES

class TestLoadStructuredDocument(object):
  def test_should_call_load_plain_file_list_if_file(self):
    with patch.object(structured_document_loader, 'load_lxml_structured_document') as mock:
      result = load_structured_document('file.lxml')
      mock.assert_called_with('file.lxml')
      assert result == mock.return_value

  def test_should_call_load_csv_or_tsv_file_list_if_file(self):
    with patch.object(structured_document_loader, 'load_svg_pages_structured_document') as mock:
      result = load_structured_document('file.svg.zip')
      mock.assert_called_with('file.svg.zip')
      assert result == mock.return_value

class TestLoadStructuredDocumentsFromFileList(object):
  def test_should_single_file(self):
    with patch.object(structured_document_loader, 'load_structured_document') as mock:
      assert (
        list(load_structured_documents_from_file_list([FILE_1])) ==
        [mock.return_value]
      )
      mock.assert_called_with(FILE_1)

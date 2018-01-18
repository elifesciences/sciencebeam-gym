from __future__ import absolute_import

from mock import patch, ANY

from lxml.builder import E

from sciencebeam_gym.structured_document.lxml import (
  LxmlStructuredDocument
)

from sciencebeam_gym.structured_document.svg import (
  SvgStructuredDocument
)

import sciencebeam_gym.structured_document.structured_document_saver as structured_document_saver
from sciencebeam_gym.structured_document.structured_document_saver import (
  save_lxml_structured_document,
  save_svg_structured_document,
  save_structured_document
)

FILE_1 = 'file1'

class TestSaveLxmlStructuredDocument(object):
  def test_should_call_save_file_content(self):
    m = structured_document_saver
    root = E.DOCUMENT()
    with patch.object(m, 'save_file_content') as save_file_content:
      with patch.object(m, 'etree') as etree:
        save_lxml_structured_document(FILE_1, LxmlStructuredDocument(root))
        save_file_content.assert_called_with(FILE_1, etree.tostring(root))

class TestSaveSvgStructuredDocument(object):
  def test_should_call_save_pages(self):
    m = structured_document_saver
    root = E.svg()
    with patch.object(m, 'save_pages') as save_pages:
      with patch.object(m, 'etree') as etree:
        save_svg_structured_document(FILE_1, SvgStructuredDocument(root))
        save_pages.assert_called_with(FILE_1, '.svg', ANY)
        args, _ = save_pages.call_args
        assert list(args[2]) == [etree.tostring(root)]

class TestSaveStructuredDocument(object):
  def test_should_call_save_lxml_structured_document(self):
    structured_document = LxmlStructuredDocument(E.DOCUMENT)
    m = structured_document_saver
    with patch.object(m, 'save_lxml_structured_document') as save_lxml_structured_document_mock:
      save_structured_document(FILE_1, structured_document)
      save_lxml_structured_document_mock.assert_called_with(FILE_1, structured_document)

  def test_should_call_save_svg_structured_document(self):
    structured_document = SvgStructuredDocument(E.svg)
    m = structured_document_saver
    with patch.object(m, 'save_svg_structured_document') as save_svg_structured_document_mock:
      save_structured_document(FILE_1, structured_document)
      save_svg_structured_document_mock.assert_called_with(FILE_1, structured_document)

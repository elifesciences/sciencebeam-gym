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

PAGE_RANGE = (2, 3)


class TestLoadLxmlStructuredDocument(object):
    def test_should_load_file(self):
        lxml_content = etree.tostring(E.test('test'))
        with NamedTemporaryFile() as f:
            f.write(lxml_content)
            f.flush()
            structured_document = load_lxml_structured_document(f.name)
            assert etree.tostring(structured_document.root) == lxml_content
            assert hasattr(structured_document.root, 'attrib')

    def test_should_limit_page_range(self):
        lxml_content = etree.tostring(E.DOCUMENT(
            E.PAGE('page 1'),
            E.PAGE('page 2'),
            E.PAGE('page 3'),
            E.PAGE('page 4')
        ))
        with NamedTemporaryFile() as f:
            f.write(lxml_content)
            f.flush()
            structured_document = load_lxml_structured_document(f.name, page_range=(2, 3))
            assert [x.text for x in structured_document.get_pages()] == ['page 2', 'page 3']


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
            assert hasattr(structured_document.page_roots[0], 'attrib')

    def test_should_limit_page_range(self):
        svg_pages_content = [
            etree.tostring(E.svg('page 1')),
            etree.tostring(E.svg('page 2')),
            etree.tostring(E.svg('page 3')),
            etree.tostring(E.svg('page 4'))
        ]
        with NamedTemporaryFile() as f:
            with ZipFile(f, 'w') as zf:
                for i, svg_page_content in enumerate(svg_pages_content):
                    zf.writestr('page-%d.svg' % (1 + i), svg_page_content)
            f.flush()
            structured_document = load_svg_pages_structured_document(f.name, page_range=(2, 3))
            assert (
                [x.text for x in structured_document.page_roots] ==
                ['page 2', 'page 3']
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
            result = load_structured_document('file.lxml', page_range=PAGE_RANGE)
            mock.assert_called_with('file.lxml', page_range=PAGE_RANGE)
            assert result == mock.return_value

    def test_should_call_load_csv_or_tsv_file_list_if_file(self):
        with patch.object(structured_document_loader, 'load_svg_pages_structured_document') as mock:
            result = load_structured_document('file.svg.zip', page_range=PAGE_RANGE)
            mock.assert_called_with('file.svg.zip', page_range=PAGE_RANGE)
            assert result == mock.return_value


class TestLoadStructuredDocumentsFromFileList(object):
    def test_should_single_file(self):
        with patch.object(structured_document_loader, 'load_structured_document') as mock:
            assert (
                list(load_structured_documents_from_file_list([FILE_1], page_range=PAGE_RANGE)) ==
                [mock.return_value]
            )
            mock.assert_called_with(FILE_1, page_range=PAGE_RANGE)

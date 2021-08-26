from subprocess import PIPE
from contextlib import contextmanager
from unittest.mock import patch

from sciencebeam_gym.pdf.pdf_to_png import (
    PdfToPng
)

from sciencebeam_gym.pdf import pdf_to_png


TEMP_DIR = '/tmp/1'

PDF_CONTENT_1 = b'pdf content 1'

ARGS_PREFIX = ['pdftoppm', '-png']
ARGS_SUFFIX = ['-', TEMP_DIR + '/page']
DEFAULT_KWARGS = dict(stdout=PIPE, stdin=PIPE, stderr=PIPE)


@contextmanager
def patch_popen():
    with patch.object(pdf_to_png, 'Popen') as mock:
        p = mock.return_value
        p.communicate.return_value = (None, None)
        p.returncode = 0
        yield mock


@contextmanager
def mock_temp_dir():
    with patch.object(pdf_to_png, 'TemporaryDirectory') as mock:
        mock.return_value.__enter__.return_value = TEMP_DIR
        with patch('os.listdir') as listdir:
            listdir.return_value = []
            yield mock


class TestPdfToPng(object):
    def test_should_pass_default_args_to_Popen(self):
        with patch_popen() as mock:
            with mock_temp_dir():
                list(PdfToPng().iter_pdf_bytes_to_png_fp(PDF_CONTENT_1))
                assert mock.called
                mock.assert_called_with(
                    ARGS_PREFIX + ARGS_SUFFIX, **DEFAULT_KWARGS
                )

    def test_should_add_page_range_to_args(self):
        with patch_popen() as mock:
            with mock_temp_dir():
                list(PdfToPng(page_range=(1, 3)).iter_pdf_bytes_to_png_fp(PDF_CONTENT_1))
                mock.assert_called_with(
                    ARGS_PREFIX + ['-f', '1', '-l', '3'] + ARGS_SUFFIX, **DEFAULT_KWARGS
                )

    def test_should_add_image_size_to_args(self):
        with patch_popen() as mock:
            with mock_temp_dir():
                list(PdfToPng(image_size=(100, 200)).iter_pdf_bytes_to_png_fp(PDF_CONTENT_1))
                mock.assert_called_with(
                    ARGS_PREFIX + ['-scale-to-x', '100', '-scale-to-y', '200'] + ARGS_SUFFIX,
                    **DEFAULT_KWARGS
                )

    def test_should_add_dpi_to_args(self):
        with patch_popen() as mock:
            with mock_temp_dir():
                list(PdfToPng(dpi=200).iter_pdf_bytes_to_png_fp(PDF_CONTENT_1))
                mock.assert_called_with(
                    ARGS_PREFIX + ['-r', '200'] + ARGS_SUFFIX, **DEFAULT_KWARGS
                )

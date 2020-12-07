import logging
from pathlib import Path

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_gym.preprocess.preprocessing_pipeline import (
    run
)

from ..example_data import EXAMPLE_DATA_DIR


LOGGER = logging.getLogger(__name__)


@pytest.mark.slow
class TestRunEndToEnd:
    def test_should_be_able_to_process_pdf(self, tmp_path: Path):
        data_path = tmp_path / 'data'
        output_path = tmp_path / 'output'
        sample_data_path = data_path / 'sample1'
        sample_data_path.mkdir(parents=True)
        sample_pdf_file_path = sample_data_path / 'sample1.pdf'
        sample_xml_file_path = sample_data_path / 'sample1.xml'
        source_pdf_file_path = Path(EXAMPLE_DATA_DIR) / 'minimal-example.pdf'
        sample_pdf_file_path.write_bytes(source_pdf_file_path.read_bytes())
        sample_xml_file_path.write_bytes(etree.tostring(E.article()))
        sample_output_path = output_path / 'sample1'
        lxml_file_path = sample_output_path / 'sample1.lxml.gz'
        svg_zip_path = sample_output_path / 'sample1.svg.zip'
        run([
            '--data-path=%s' % data_path,
            '--output-path=%s' % output_path,
            '--pdf-path=%s' % sample_pdf_file_path,
            '--xml-path=%s' % sample_xml_file_path,
            '--save-lxml',
            '--save-svg',
            '--save-tfrecords'
        ])
        assert lxml_file_path.exists()
        assert svg_zip_path.exists()
        tfrecords_file_paths = list(output_path.glob('*.tfrecord.gz'))
        LOGGER.debug('tfrecords_file_paths: %s', tfrecords_file_paths)
        assert tfrecords_file_paths

    def test_should_be_able_to_process_lxml(self, tmp_path: Path):
        data_path = tmp_path / 'data'
        data_path.mkdir()
        lxml_file_path = tmp_path / 'test.lxml'
        lxml_file_path.write_bytes(etree.tostring(E.lxml()))
        xml_file_path = tmp_path / 'test.xml'
        xml_file_path.write_bytes(etree.tostring(E.article()))
        svg_zip_path = tmp_path / 'test.svg.zip'
        run([
            '--data-path=%s' % data_path,
            '--lxml-path=%s' % lxml_file_path,
            '--xml-path=%s' % xml_file_path,
            '--save-svg'
        ])
        assert svg_zip_path.exists()

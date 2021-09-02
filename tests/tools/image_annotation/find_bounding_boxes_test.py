import gzip
import json
import logging
from io import BytesIO
from pathlib import Path

import PIL.Image
import pytest
import numpy as np
from lxml import etree
from lxml.builder import ElementMaker
from sklearn.datasets import load_sample_image

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.utils.cv import (
    copy_image_to
)
from sciencebeam_gym.tools.image_annotation.find_bounding_boxes import (
    XLINK_NS,
    XLINK_HREF,
    main
)


LOGGER = logging.getLogger(__name__)


JATS_E = ElementMaker(nsmap={
    'xlink': XLINK_NS
})


@pytest.fixture(name='sample_image_array', scope='session')
def _sample_image_array() -> np.ndarray:
    return load_sample_image('flower.jpg')


@pytest.fixture(name='sample_image', scope='session')
def _sample_image(sample_image_array: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(sample_image_array)


class TestMain:
    def test_should_annotate_single_full_page_image(
        self,
        tmp_path: Path,
        sample_image: PIL.Image.Image
    ):
        LOGGER.debug('sample_image: %sx%s', sample_image.width, sample_image.height)
        image_path = tmp_path / 'test.jpg'
        pdf_path = tmp_path / 'test.pdf'
        output_json_path = tmp_path / 'test.json'
        sample_image.save(image_path, 'JPEG')
        sample_image.save(
            pdf_path,
            'PDF',
            resolution=100.0,
            save_all=True,
            append_images=[]
        )
        main([
            '--pdf-file',
            str(pdf_path),
            '--image-files',
            str(image_path),
            '--output-json-file',
            str(output_json_path)
        ])
        assert output_json_path.exists()
        json_data = json.loads(output_json_path.read_text())
        LOGGER.debug('json_data: %s', json_data)
        images_json = json_data['images']
        assert len(images_json) == 1
        image_json = images_json[0]
        assert image_json['width'] == 1280
        assert image_json['height'] == 854
        categories_json = json_data['categories']
        assert len(categories_json) == 1
        assert categories_json[0]['name'] == 'figure'
        assert categories_json[0]['id'] == 1
        annotations_json = json_data['annotations']
        assert len(annotations_json) == 1
        annotation_json = annotations_json[0]
        assert annotation_json['image_id'] == image_json['id']
        assert annotation_json['category_id'] == categories_json[0]['id']
        assert annotation_json['bbox'] == [
            0, 0, image_json['width'], image_json['height']
        ]

    def test_should_annotate_smaller_image(
        self,
        tmp_path: Path,
        sample_image: PIL.Image.Image
    ):
        LOGGER.debug('sample_image: %sx%s', sample_image.width, sample_image.height)
        image_path = tmp_path / 'test.jpg'
        pdf_path = tmp_path / 'test.pdf'
        output_json_path = tmp_path / 'test.json'
        pdf_page_image = np.full((400, 600, 3), 255, dtype=np.uint8)
        copy_image_to(
            np.asarray(sample_image),
            pdf_page_image,
            BoundingBox(20, 30, 240, 250),
        )
        sample_image.save(image_path, 'JPEG')
        PIL.Image.fromarray(pdf_page_image).save(
            pdf_path,
            'PDF',
            resolution=100.0,
            save_all=True,
            append_images=[]
        )
        main([
            '--pdf-file',
            str(pdf_path),
            '--image-files',
            str(image_path),
            '--output-json-file',
            str(output_json_path)
        ])
        assert output_json_path.exists()
        json_data = json.loads(output_json_path.read_text())
        LOGGER.debug('json_data: %s', json_data)
        images_json = json_data['images']
        assert len(images_json) == 1
        image_json = images_json[0]
        assert image_json['width'] == 1200
        assert image_json['height'] == 800
        annotations_json = json_data['annotations']
        annotation_json = annotations_json[0]
        np.testing.assert_allclose(
            annotation_json['bbox'],
            [40, 60, 480, 500],
            atol=10
        )

    def test_should_annotate_using_jats_xml(
        self,
        tmp_path: Path,
        sample_image: PIL.Image.Image
    ):
        LOGGER.debug('sample_image: %sx%s', sample_image.width, sample_image.height)
        image_path = tmp_path / 'test.jpg'
        pdf_path = tmp_path / 'test.pdf'
        xml_path = tmp_path / 'test.xml'
        xml_path.write_bytes(etree.tostring(
            JATS_E.article(JATS_E.body(JATS_E.sec(JATS_E.fig(
                JATS_E.graphic({XLINK_HREF: image_path.name})
            ))))
        ))
        output_json_path = tmp_path / 'test.json'
        sample_image.save(image_path, 'JPEG')
        sample_image.save(
            pdf_path,
            'PDF',
            resolution=100.0,
            save_all=True,
            append_images=[]
        )
        main([
            '--pdf-file',
            str(pdf_path),
            '--xml-file',
            str(xml_path),
            '--output-json-file',
            str(output_json_path)
        ])
        assert output_json_path.exists()
        json_data = json.loads(output_json_path.read_text())
        LOGGER.debug('json_data: %s', json_data)
        images_json = json_data['images']
        assert len(images_json) == 1
        image_json = images_json[0]
        assert image_json['width'] == 1280
        assert image_json['height'] == 854
        categories_json = json_data['categories']
        assert len(categories_json) == 1
        assert categories_json[0]['name'] == 'figure'
        assert categories_json[0]['id'] == 1
        annotations_json = json_data['annotations']
        assert len(annotations_json) == 1
        annotation_json = annotations_json[0]
        assert annotation_json['image_id'] == image_json['id']
        assert annotation_json['category_id'] == categories_json[0]['id']
        assert annotation_json['bbox'] == [
            0, 0, image_json['width'], image_json['height']
        ]
        assert annotation_json['file_name'] == image_path.name

    def test_should_annotate_using_jats_xml_and_gzipped_files(
        self,
        tmp_path: Path,
        sample_image: PIL.Image.Image
    ):
        LOGGER.debug('sample_image: %sx%s', sample_image.width, sample_image.height)
        image_path = tmp_path / 'test.jpg.gz'
        pdf_path = tmp_path / 'test.pdf.gz'
        xml_path = tmp_path / 'test.xml.gz'
        xml_path.write_bytes(gzip.compress(etree.tostring(
            JATS_E.article(JATS_E.body(JATS_E.sec(JATS_E.fig(
                JATS_E.graphic({XLINK_HREF: 'test.jpg'})
            ))))
        )))
        output_json_path = tmp_path / 'test.json'

        temp_out = BytesIO()
        sample_image.save(temp_out, 'JPEG')
        image_path.write_bytes(gzip.compress(temp_out.getvalue()))

        temp_out = BytesIO()
        sample_image.save(
            temp_out,
            'PDF',
            resolution=100.0,
            save_all=True,
            append_images=[]
        )
        pdf_path.write_bytes(gzip.compress(temp_out.getvalue()))
        main([
            '--pdf-file',
            str(pdf_path),
            '--xml-file',
            str(xml_path),
            '--output-json-file',
            str(output_json_path)
        ])
        assert output_json_path.exists()
        json_data = json.loads(output_json_path.read_text())
        LOGGER.debug('json_data: %s', json_data)
        images_json = json_data['images']
        assert len(images_json) == 1
        image_json = images_json[0]
        assert image_json['width'] == 1280
        assert image_json['height'] == 854
        categories_json = json_data['categories']
        assert len(categories_json) == 1
        assert categories_json[0]['name'] == 'figure'
        assert categories_json[0]['id'] == 1
        annotations_json = json_data['annotations']
        assert len(annotations_json) == 1
        annotation_json = annotations_json[0]
        assert annotation_json['image_id'] == image_json['id']
        assert annotation_json['category_id'] == categories_json[0]['id']
        assert annotation_json['bbox'] == [
            0, 0, image_json['width'], image_json['height']
        ]

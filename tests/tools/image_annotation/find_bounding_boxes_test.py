import gzip
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import IO, List, Union

import PIL.Image
import pytest
import numpy as np
from lxml import etree
from lxml.builder import ElementMaker
from sklearn.datasets import load_sample_image

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.utils.cv import (
    resize_image,
    copy_image_to
)
from sciencebeam_gym.tools.image_annotation.find_bounding_boxes import (
    XLINK_NS,
    XLINK_HREF,
    CategoryNames,
    GraphicImageNotFoundError,
    main
)


LOGGER = logging.getLogger(__name__)


JATS_E = ElementMaker(nsmap={
    'xlink': XLINK_NS
})


SAMPLE_IMAGE_WIDTH = 320
SAMPLE_IMAGE_HEIGHT = 240

SAMPLE_PDF_PAGE_WIDTH = SAMPLE_IMAGE_WIDTH * 2
SAMPLE_PDF_PAGE_HEIGHT = SAMPLE_IMAGE_HEIGHT * 2


@pytest.fixture(name='sample_image_array', scope='session')
def _sample_image_array() -> np.ndarray:
    return resize_image(
        load_sample_image('flower.jpg'),
        width=SAMPLE_IMAGE_WIDTH,
        height=SAMPLE_IMAGE_HEIGHT
    )


@pytest.fixture(name='sample_image', scope='session')
def _sample_image(sample_image_array: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(sample_image_array)


@pytest.fixture(name='sample_image_array_2', scope='session')
def _sample_image_array_2() -> np.ndarray:
    return resize_image(
        load_sample_image('china.jpg'),
        width=SAMPLE_IMAGE_WIDTH,
        height=SAMPLE_IMAGE_HEIGHT
    )


@pytest.fixture(name='sample_image_2', scope='session')
def _sample_image_2(sample_image_array_2: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(sample_image_array_2)


def save_images_as_pdf(path_or_io: Union[str, Path, IO], images: List[PIL.Image.Image]):
    images[0].save(
        path_or_io,
        'PDF',
        resolution=100.0,
        save_all=True,
        append_images=images[1:]
    )


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
        save_images_as_pdf(pdf_path, [sample_image])
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
        assert image_json['width'] == SAMPLE_PDF_PAGE_WIDTH
        assert image_json['height'] == SAMPLE_PDF_PAGE_HEIGHT
        categories_json = json_data['categories']
        assert len(categories_json) == 1
        assert categories_json[0]['name'] == CategoryNames.UNKNOWN_GRAPHIC
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
        save_images_as_pdf(pdf_path, [PIL.Image.fromarray(pdf_page_image)])
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
        save_images_as_pdf(pdf_path, [sample_image])
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
        assert image_json['width'] == SAMPLE_PDF_PAGE_WIDTH
        assert image_json['height'] == SAMPLE_PDF_PAGE_HEIGHT
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

    def test_should_annotate_multiple_images_using_jats_xml(
        self,
        tmp_path: Path,
        sample_image: PIL.Image.Image,
        sample_image_2: PIL.Image.Image
    ):
        LOGGER.debug('sample_image: %sx%s', sample_image.width, sample_image.height)
        image_path = tmp_path / 'test.jpg'
        image_2_path = tmp_path / 'test2.jpg'
        pdf_path = tmp_path / 'test.pdf'
        xml_path = tmp_path / 'test.xml'
        xml_path.write_bytes(etree.tostring(
            JATS_E.article(JATS_E.body(JATS_E.sec(
                JATS_E.fig(
                    {'id': 'fig1'},
                    JATS_E.graphic({XLINK_HREF: image_path.name})
                ),
                JATS_E('table-wrap', *[
                    {'id': 'tab1'},
                    JATS_E.graphic({XLINK_HREF: image_2_path.name})
                ])
            )))
        ))
        output_json_path = tmp_path / 'test.json'
        sample_image.save(image_path, 'JPEG')
        sample_image_2.save(image_2_path, 'JPEG')
        save_images_as_pdf(pdf_path, [sample_image, sample_image_2])
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
        assert len(images_json) == 2
        categories_json = json_data['categories']
        category_names = [c['name'] for c in categories_json]
        assert category_names == [
            CategoryNames.FIGURE,
            CategoryNames.TABLE
        ]
        annotations_json = json_data['annotations']
        assert len(annotations_json) == 2

        assert annotations_json[0]['image_id'] == images_json[0]['id']
        assert annotations_json[0]['category_id'] == categories_json[0]['id']
        assert annotations_json[0]['bbox'] == [
            0, 0, images_json[0]['width'], images_json[0]['height']
        ]
        assert annotations_json[0]['file_name'] == image_path.name
        assert annotations_json[0]['related_element_id'] == 'fig1'

        assert annotations_json[1]['image_id'] == images_json[1]['id']
        assert annotations_json[1]['category_id'] == categories_json[1]['id']
        assert annotations_json[1]['bbox'] == [
            0, 0, images_json[1]['width'], images_json[1]['height']
        ]
        assert annotations_json[1]['file_name'] == image_2_path.name
        assert annotations_json[1]['related_element_id'] == 'tab1'

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
        save_images_as_pdf(temp_out, [sample_image])
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
        assert image_json['width'] == SAMPLE_PDF_PAGE_WIDTH
        assert image_json['height'] == SAMPLE_PDF_PAGE_HEIGHT
        categories_json = json_data['categories']
        assert len(categories_json) == 1
        assert categories_json[0]['name'] == CategoryNames.FIGURE
        assert categories_json[0]['id'] == 1
        annotations_json = json_data['annotations']
        assert len(annotations_json) == 1
        annotation_json = annotations_json[0]
        assert annotation_json['image_id'] == image_json['id']
        assert annotation_json['category_id'] == categories_json[0]['id']
        assert annotation_json['bbox'] == [
            0, 0, image_json['width'], image_json['height']
        ]

    def test_should_raise_error_when_image_could_not_be_found(
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
        save_images_as_pdf(pdf_path, [PIL.Image.fromarray(np.zeros((200, 200), dtype=np.uint8))])
        with pytest.raises(GraphicImageNotFoundError):
            main([
                '--pdf-file',
                str(pdf_path),
                '--xml-file',
                str(xml_path),
                '--output-json-file',
                str(output_json_path)
            ])

    def test_should_output_missing_annotations(
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
        save_images_as_pdf(pdf_path, [PIL.Image.fromarray(np.zeros((200, 200), dtype=np.uint8))])
        main([
            '--pdf-file',
            str(pdf_path),
            '--xml-file',
            str(xml_path),
            '--output-json-file',
            str(output_json_path),
            '--skip-errors'
        ])
        assert output_json_path.exists()
        json_data = json.loads(output_json_path.read_text())
        LOGGER.debug('json_data: %s', json_data)
        missing_annotations_json = json_data['missing_annotations']
        assert [a['file_name'] for a in missing_annotations_json] == [image_path.name]

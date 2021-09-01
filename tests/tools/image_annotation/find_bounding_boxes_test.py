import json
import logging
from pathlib import Path

import PIL.Image
import numpy as np
from cv2 import cv2 as cv
from lxml import etree
from lxml.builder import ElementMaker
from sklearn.datasets import load_sample_image

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.tools.image_annotation.find_bounding_boxes import (
    XLINK_NS,
    XLINK_HREF,
    main
)


LOGGER = logging.getLogger(__name__)


JATS_E = ElementMaker(nsmap={
    'xlink': XLINK_NS
})


def resize_image(src: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv.resize(
        src,
        dsize=(width, height),
        interpolation=cv.INTER_CUBIC
    )


def copy_image_to(
    src: np.ndarray,
    dst: np.ndarray,
    dst_bounding_box: BoundingBox,
):
    x = int(dst_bounding_box.x)
    y = int(dst_bounding_box.y)
    width = int(dst_bounding_box.width)
    height = int(dst_bounding_box.height)
    dst[y:(y + height), x:(x + width)] = resize_image(
        src, width=width, height=height
    )


class TestMain:
    def test_should_annotate_single_full_page_image(self, tmp_path: Path):
        sample_image = load_sample_image('flower.jpg')
        LOGGER.debug('sample_image: %s (%s)', sample_image.shape, sample_image.dtype)
        image_path = tmp_path / 'test.jpg'
        pdf_path = tmp_path / 'test.pdf'
        output_json_path = tmp_path / 'test.json'
        pil_image = PIL.Image.fromarray(sample_image)
        pil_image.save(image_path, 'JPEG')
        pil_image.save(
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

    def test_should_annotate_smaller_image(self, tmp_path: Path):
        sample_image = load_sample_image('flower.jpg')
        LOGGER.debug('sample_image: %s (%s)', sample_image.shape, sample_image.dtype)
        image_path = tmp_path / 'test.jpg'
        pdf_path = tmp_path / 'test.pdf'
        output_json_path = tmp_path / 'test.json'
        pdf_page_image = np.zeros((400, 600, 3), dtype=np.uint8)
        copy_image_to(
            sample_image,
            pdf_page_image,
            BoundingBox(20, 30, 240, 250),
        )
        PIL.Image.fromarray(sample_image).save(image_path, 'JPEG')
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

    def test_should_annotate_using_jats_xml(self, tmp_path: Path):
        sample_image = load_sample_image('flower.jpg')
        LOGGER.debug('sample_image: %s (%s)', sample_image.shape, sample_image.dtype)
        image_path = tmp_path / 'test.jpg'
        pdf_path = tmp_path / 'test.pdf'
        xml_path = tmp_path / 'test.xml'
        xml_path.write_bytes(etree.tostring(
            JATS_E.article(JATS_E.body(JATS_E.sec(JATS_E.fig(
                JATS_E.graphic({XLINK_HREF: image_path.name})
            ))))
        ))
        output_json_path = tmp_path / 'test.json'
        pil_image = PIL.Image.fromarray(sample_image)
        pil_image.save(image_path, 'JPEG')
        pil_image.save(
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

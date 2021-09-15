import gzip
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import IO, List, Union

import dill
import PIL.Image
import pytest
import numpy as np
from lxml import etree
from lxml.builder import ElementMaker
from sciencebeam_utils.utils.file_path import relative_path
from sklearn.datasets import load_sample_image

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.utils.cv import (
    resize_image,
    copy_image_to
)
from sciencebeam_gym.tools.image_annotation.find_bounding_boxes_utils import (
    COORDS_ATTRIB_NAME,
    COORDS_NS_NAMEMAP,
    XLINK_NS,
    XLINK_HREF,
    CategoryNames,
    FindBoundingBoxPipelineFactory,
    GraphicImageNotFoundError,
    format_coords_attribute_value,
    main,
    parse_args,
    save_annotated_images
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


class TestSaveAnnotatedImages:
    def test_should_not_fail(
        self,
        tmp_path: Path,
        sample_image: PIL.Image.Image,

    ):
        save_annotated_images(
            pdf_images=[sample_image],
            annotations=[{
                'image_id': 1,
                'category_id': 1,
                'file_name': 'sample.jpg',
                'bbox': BoundingBox(
                    10, 10, sample_image.width - 10, sample_image.height - 10
                ).to_list()
            }],
            output_annotated_images_path=str(tmp_path),
            category_name_by_id={1: 'figure'}
        )


class TestFindBoundingBoxPipelineFactory:
    def test_should_be_able_to_serialize(self, tmp_path: Path):
        args = parse_args([
            '--pdf-file-list=pdf-file-list1',
            '--xml-file-list=xml-file-list1',
            '--output-json-file=output-json-file1'
            '--resume'
        ])
        pipeline_factory = FindBoundingBoxPipelineFactory(args)
        dill.loads(dill.dumps(pipeline_factory))
        dill.loads(dill.dumps(pipeline_factory.process_item))
        dill.dump_session(str(tmp_path / 'session.pkl'))


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

    def test_should_annotate_using_jats_xml_from_tsv_file_list(
        self,
        tmp_path: Path,
        sample_image: PIL.Image.Image
    ):
        LOGGER.debug('sample_image: %sx%s', sample_image.width, sample_image.height)
        source_path = tmp_path / 'source'
        article_source_path = source_path / 'article1'
        article_source_path.mkdir(parents=True)
        image_path = article_source_path / 'test.jpg'
        pdf_path = article_source_path / 'test.pdf'
        xml_path = article_source_path / 'test.xml'
        file_list_path = source_path / 'file-list.tsv'
        file_list_path.write_text('\n'.join([
            '\t'.join(['source_url', 'xml_url']),
            '\t'.join([str(pdf_path), str(xml_path)])
        ]))
        xml_root = JATS_E.article(JATS_E.body(JATS_E.sec(JATS_E.fig(
            JATS_E.graphic({XLINK_HREF: image_path.name})
        ))))
        xml_path.write_bytes(etree.tostring(xml_root))
        output_path = tmp_path / 'output'
        article_output_path = output_path / article_source_path.name
        images_output_path = article_output_path / 'images'
        output_json_path = article_output_path / 'json' / 'test.json'
        output_xml_path = article_output_path / 'xml' / 'test.xml'
        sample_image.save(image_path, 'JPEG')
        save_images_as_pdf(pdf_path, [sample_image])
        main([
            '--pdf-file-list',
            str(file_list_path),
            '--pdf-file-column=source_url',
            '--pdf-base-path',
            str(source_path),
            '--xml-file-list',
            str(file_list_path),
            '--xml-file-column=xml_url',
            '--output-path',
            str(output_path),
            '--output-json-file',
            relative_path(str(article_output_path), str(output_json_path)),
            '--output-xml-file',
            relative_path(str(article_output_path), str(output_xml_path)),
            '--output-annotated-images-path',
            relative_path(str(article_output_path), str(images_output_path))
        ])
        assert output_json_path.exists()
        assert images_output_path.exists()
        assert images_output_path.glob('*.png')
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
        assert output_xml_path.exists()
        output_xml_root = etree.fromstring(output_xml_path.read_bytes())
        LOGGER.debug('output_xml_root: %r', etree.tostring(output_xml_root))
        output_graphic_element = output_xml_root.xpath('//fig/graphic')[0]
        assert output_graphic_element.get(COORDS_ATTRIB_NAME) == (
            format_coords_attribute_value(
                page_number=1,
                bounding_box=BoundingBox(*annotation_json['bbox'])
            )
        )
        assert output_xml_root.nsmap == {**xml_root.nsmap, **COORDS_NS_NAMEMAP}

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

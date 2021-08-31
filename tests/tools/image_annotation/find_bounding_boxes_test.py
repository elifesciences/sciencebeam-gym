import json
import logging
from pathlib import Path

import PIL.Image
from sklearn.datasets import load_sample_image

from sciencebeam_gym.tools.image_annotation.find_bounding_boxes import main


LOGGER = logging.getLogger(__name__)


class TestMain:
    def test_should_annotate_image(self, tmp_path: Path):
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
            '--image-file',
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
        # assert False

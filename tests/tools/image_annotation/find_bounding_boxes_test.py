import logging
from pathlib import Path

import PIL.Image
from sklearn.datasets import load_sample_image

from sciencebeam_gym.tools.image_annotation.find_bounding_boxes import main


LOGGER = logging.getLogger(__name__)


class TestMain:
    def test_should_annotate_image(self, tmp_path: Path):
        sample_image = load_sample_image('flower.jpg')
        LOGGER.debug('flower_image: %s (%s)', sample_image.shape, sample_image.dtype)
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
            pdf_path,
            '--image-file',
            image_path,
            '--output-json-file',
            output_json_path
        ])
        # assert output_json_path.exists()

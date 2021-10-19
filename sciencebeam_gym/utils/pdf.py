from typing import List, Tuple

import pdfminer.pdfparser
import pdfminer.pdfdocument
import pdfminer.pdfpage

from sciencebeam_gym.utils.bounding_box import BoundingBox


def get_bounding_box_for_x1_y1_x2_y2(
    x1_y1_x2_y2: Tuple[float, float, float, float]
) -> BoundingBox:
    return BoundingBox(
        x=x1_y1_x2_y2[0],
        y=x1_y1_x2_y2[1],
        width=x1_y1_x2_y2[2] - x1_y1_x2_y2[0],
        height=x1_y1_x2_y2[3] - x1_y1_x2_y2[1]
    )


def get_page_bounding_boxes_from_local_pdf(
    local_pdf_path: str
) -> List[BoundingBox]:
    with open(local_pdf_path, 'rb') as pdf_fp:
        parser = pdfminer.pdfparser.PDFParser(pdf_fp)
        doc = pdfminer.pdfdocument.PDFDocument(parser)
        return [
            get_bounding_box_for_x1_y1_x2_y2(page.mediabox)
            for page in pdfminer.pdfpage.PDFPage.create_pages(doc)
        ]

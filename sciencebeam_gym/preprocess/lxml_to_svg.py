import argparse
import logging
import os

from lxml import etree

from sciencebeam_gym.utils.bounding_box import (
  BoundingBox
)

from sciencebeam_gym.utils.csv import (
  open_csv_output,
  write_dict_csv
)

from sciencebeam_gym.preprocess.annotation.annotator import (
  Annotator,
  DEFAULT_ANNOTATORS
)

from sciencebeam_gym.preprocess.annotation.matching_annotator import (
  MatchingAnnotator,
  CsvMatchDetailReporter
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
  parse_xml_mapping,
  xml_root_to_target_annotations
)

from sciencebeam_gym.preprocess.annotation.annotation_evaluation import (
  evaluate_document_by_page,
  DEFAULT_EVALUATION_COLUMNS,
  to_csv_dict_rows as to_annotation_evaluation_csv_dict_rows
)

from sciencebeam_gym.structured_document.svg import (
  SVG_TEXT,
  SVG_G,
  SVG_RECT,
  SVG_DOC,
  SVG_NSMAP,
  SVGE_BOUNDING_BOX,
  SvgStructuredDocument,
  SvgStyleClasses,
  format_bounding_box as svg_format_bounding_box
)

from sciencebeam_gym.preprocess.visualize_svg_annotation import (
  visualize_svg_annotations
)

def get_logger():
  return logging.getLogger(__name__)

def ElementWithText(tag, text, **kwargs):
  node = etree.Element(tag, **kwargs)
  node.text = text
  return node

def svg_pattern_for_lxml_path(lxml_path):
  name, _ = os.path.splitext(lxml_path)
  return name + '-page{}.svg'

def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description='Convert to LXML (pdftoxml) to SVG'
  )
  parser.add_argument(
    '--lxml-path', type=str, required=True,
    help='path to lxml file'
  )
  parser.add_argument(
    '--svg-path', type=str, required=False,
    help='path to svg file'
  )
  parser.add_argument(
    '--xml-path', type=str, required=False,
    help='path to xml file'
  )
  parser.add_argument(
    '--xml-mapping-path', type=str, default='annot-xml-front.conf',
    help='path to xml mapping file'
  )
  parser.add_argument(
    '--annotate', action='store_true', required=False,
    help='enable annotation'
  )
  parser.add_argument(
    '--debug', action='store_true', required=False,
    help='enable debug logging'
  )
  parser.add_argument(
    '--debug-match', type=str, required=False,
    help='debug matches and save as csv'
  )
  parser.add_argument(
    '--annotation-evaluation-csv', type=str, required=False,
    help='Annotation evaluation CSV output file'
  )
  args = parser.parse_args(argv)
  return args

def iter_svg_pages_for_lxml(lxml_root, add_background=True):
  previous_block = None
  previous_svg_block = None
  for page in lxml_root.xpath('//DOCUMENT/PAGE'):
    svg_root = etree.Element(SVG_DOC, nsmap=SVG_NSMAP)
    page_width = page.attrib.get('width')
    page_height = page.attrib.get('height')
    if page_width and page_height:
      svg_root.attrib['viewBox'] = '0 0 %s %s' % (page_width, page_height)
    if add_background:
      svg_root.append(etree.Element(SVG_RECT, attrib={
        'width': '100%',
        'height': '100%',
        'fill': 'white',
        'class': 'background'
      }))
    for text in page.xpath('.//TEXT'):
      svg_g = etree.Element(SVG_G, nsmap=SVG_NSMAP, attrib={
        'class': SvgStyleClasses.LINE
      })
      for token in text.xpath('./TOKEN'):
        x = float(token.attrib.get('x'))
        y = float(token.attrib.get('y'))
        height = float(token.attrib.get('height'))
        width = float(token.attrib.get('width'))
        base = float(token.attrib.get('base', y))
        y_center = y + height / 2.0
        attrib = {
          'x': str(x),
          'y': str(base),
          'font-size': token.attrib.get('font-size'),
          'font-family': token.attrib.get('font-name'),
          'fill': token.attrib.get('font-color'),
          SVGE_BOUNDING_BOX: svg_format_bounding_box(BoundingBox(
            x, y, width, height
          ))
        }
        angle = float(token.attrib.get('angle', '0'))
        if token.attrib.get('rotation') == '1' and angle == 90.0:
          attrib['x'] = '0'
          attrib['y'] = '0'
          attrib['transform'] = 'translate({x} {y}) rotate({angle})'.format(
            x=str(x),
            y=str(y_center),
            angle=str(-angle)
          )
        svg_g.append(
          ElementWithText(SVG_TEXT, token.text, attrib=attrib)
        )
      text_parent = text.getparent()
      if text_parent.tag == 'BLOCK':
        if text_parent != previous_block:
          previous_svg_block = etree.Element(SVG_G, nsmap=SVG_NSMAP, attrib={
            'class': SvgStyleClasses.BLOCK
          })
          svg_root.append(previous_svg_block)
          previous_block = text_parent
        previous_svg_block.append(svg_g)
      else:
        previous_block = None
        previous_svg_block = None
        svg_root.append(svg_g)
    yield svg_root

def convert(args):
  logger = get_logger()
  svg_filename_pattern = args.svg_path
  if not svg_filename_pattern:
    svg_filename_pattern = svg_pattern_for_lxml_path(args.lxml_path)
  logger.debug('svg_filename_pattern: %s', svg_filename_pattern)
  lxml_root = etree.parse(args.lxml_path).getroot()

  match_detail_reporter = None
  if args.annotate:
    annotators = DEFAULT_ANNOTATORS
    if args.debug_match:
      match_detail_reporter = CsvMatchDetailReporter(
        open_csv_output(args.debug_match),
        args.debug_match
      )
    if args.xml_path:
      xml_mapping = parse_xml_mapping(args.xml_mapping_path)
      target_annotations = xml_root_to_target_annotations(
        etree.parse(args.xml_path).getroot(),
        xml_mapping
      )
      annotators = annotators + [MatchingAnnotator(
        target_annotations, match_detail_reporter=match_detail_reporter
      )]
    annotator = Annotator(annotators)
  else:
    annotator = None

  if annotator:
    svg_roots = list(iter_svg_pages_for_lxml(lxml_root))
    annotator.annotate(SvgStructuredDocument(svg_roots))
  else:
    svg_roots = iter_svg_pages_for_lxml(lxml_root)
  for page_index, svg_root in enumerate(svg_roots):
    if annotator:
      svg_root = visualize_svg_annotations(svg_root)
    svg_filename = svg_filename_pattern.format(1 + page_index)
    logger.info('writing to: %s', svg_filename)
    with open(svg_filename, 'wb') as f:
      etree.ElementTree(svg_root).write(f, pretty_print=True)
  if annotator:
    tagging_evaluation_results = evaluate_document_by_page(SvgStructuredDocument(svg_roots))
    logger.info('tagging evaluation:\n%s', '\n'.join([
      'page{}: {}'.format(1 + i, r) for i, r in enumerate(tagging_evaluation_results)
    ]))
    if args.annotation_evaluation_csv:
      write_dict_csv(
        args.annotation_evaluation_csv,
        DEFAULT_EVALUATION_COLUMNS,
        to_annotation_evaluation_csv_dict_rows(
          tagging_evaluation_results,
          document=os.path.basename(args.lxml_path)
        )
      )
  if match_detail_reporter:
    match_detail_reporter.close()

def main():
  args = parse_args()
  if args.debug:
    logging.basicConfig(level=logging.DEBUG)
  else:
    logging.basicConfig(level=logging.INFO)
  convert(args)

if __name__ == "__main__":
  main()

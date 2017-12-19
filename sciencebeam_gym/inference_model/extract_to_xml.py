import argparse
import logging
from io import BytesIO

from lxml import etree
from lxml.builder import E

from sciencebeam_gym.utils.tf import (
  FileIO
)

from sciencebeam_gym.structured_document.lxml import (
  LxmlStructuredDocument
)

from sciencebeam_gym.inference_model.extract_from_annotated_document import (
  extract_from_annotated_document
)

class Tags(object):
  TITLE = 'manuscript_title'
  ABSTRACT = 'abstract'

class XmlPaths(object):
  TITLE = 'front/article-meta/title-group/article-title'
  ABSTRACT = 'front/article-meta/abstract'

def get_logger():
  return logging.getLogger(__name__)

def rsplit_xml_path(path):
  i = path.rfind('/')
  if i >= 0:
    return path[0:i], path[i + 1:]
  else:
    return None, path

def create_node_recursive(xml_root, path, exists_ok=False):
  node = xml_root.find(path)
  if node is not None:
    if not exists_ok:
      raise RuntimeError('xml node already exists: %s' % path)
    return node
  parent, base = rsplit_xml_path(path)
  if parent:
    parent_node = create_node_recursive(xml_root, parent, exists_ok=True)
  else:
    parent_node = xml_root
  node = etree.Element(base)
  parent_node.append(node)
  return node

def set_xml_text(xml_root, path, text):
  node = create_node_recursive(xml_root, path, exists_ok=True)
  if node.text is None:
    node.text = text
  else:
    node.text += '\n' + text
  return node

def extracted_items_to_xml(extracted_items):
  simple_xml_mapping = {
    Tags.TITLE: XmlPaths.TITLE,
    Tags.ABSTRACT: XmlPaths.ABSTRACT
  }
  xml_root = E.article()
  for extracted_item in extracted_items:
    tag = extracted_item.tag
    if tag:
      path = simple_xml_mapping.get(tag)
      if not path:
        get_logger().warning('tag not configured: %s', tag)
        continue
      set_xml_text(xml_root, path, extracted_item.text)
  return xml_root

def extract_structured_document_to_xml(structured_document):
  return extracted_items_to_xml(
    extract_from_annotated_document(structured_document)
  )

def read_all(path, mode):
  with FileIO(path, mode) as f:
    return f.read()

def parse_args(argv=None):
  parser = argparse.ArgumentParser('Extract JATSy XML from annotated LXML')
  parser.add_argument(
    '--lxml-path', type=str, required=True,
    help='path to lxml document'
  )

  parser.add_argument(
    '--output-path', type=str, required=True,
    help='output path to annotated document'
  )

  parser.add_argument(
    '--debug', action='store_true', default=False,
    help='enable debug output'
  )

  return parser.parse_args(argv)

def main(argv=None):
  args = parse_args(argv)

  if args.debug:
    logging.getLogger().setLevel('DEBUG')

  structured_document = LxmlStructuredDocument(
    etree.parse(BytesIO(read_all(args.lxml_path, 'rb')))
  )

  xml_root = extract_structured_document_to_xml(structured_document)

  get_logger().info('writing result to: %s', args.output_path)
  with FileIO(args.output_path, 'w') as out_f:
    out_f.write(etree.tostring(xml_root, pretty_print=True))

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

import argparse
import logging

from lxml import etree
from lxml.builder import E

from sciencebeam_gym.beam_utils.io import (
  save_file_content
)

from sciencebeam_gym.structured_document.structured_document_loader import (
  load_structured_document
)

from sciencebeam_gym.inference_model.extract_from_annotated_document import (
  extract_from_annotated_document
)

class Tags(object):
  TITLE = 'manuscript_title'
  ABSTRACT = 'abstract'
  AUTHOR = 'author'
  AUTHOR_AFF = 'author_aff'

class XmlPaths(object):
  TITLE = 'front/article-meta/title-group/article-title'
  ABSTRACT = 'front/article-meta/abstract'
  AUTHOR = 'front/article-meta/contrib-group/contrib/name'
  AUTHOR_AFF = 'front/article-meta/contrib-group/aff'

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

def create_xml_text(xml_root, path, text):
  parent, base = rsplit_xml_path(path)
  parent_node = create_node_recursive(xml_root, parent, exists_ok=True)
  node = etree.Element(base)
  node.text = text
  parent_node.append(node)
  return node

class XmlMapping(object):
  def __init__(self, xml_path, single_node=False):
    self.xml_path = xml_path
    self.single_node = single_node

def extracted_items_to_xml(extracted_items):
  xml_mapping = {
    Tags.TITLE: XmlMapping(XmlPaths.TITLE, single_node=True),
    Tags.ABSTRACT: XmlMapping(XmlPaths.ABSTRACT, single_node=True),
    Tags.AUTHOR: XmlMapping(XmlPaths.AUTHOR),
    Tags.AUTHOR_AFF: XmlMapping(XmlPaths.AUTHOR_AFF)
  }
  xml_root = E.article()
  previous_tag = None
  for extracted_item in extracted_items:
    tag = extracted_item.tag
    if tag:
      mapping_entry = xml_mapping.get(tag)
      if not mapping_entry:
        get_logger().warning('tag not configured: %s', tag)
        continue
      path = mapping_entry.xml_path
      if mapping_entry.single_node:
        node = create_node_recursive(xml_root, path, exists_ok=True)
        if node.text is None:
          node.text = extracted_item.text
        elif previous_tag == tag:
          node.text += '\n' + extracted_item.text
        else:
          get_logger().debug('ignoring tag %s, after tag %s', tag, previous_tag)
      else:
        create_xml_text(xml_root, path, extracted_item.text)
      previous_tag = tag
  return xml_root

def extract_structured_document_to_xml(structured_document, tag_scope=None):
  return extracted_items_to_xml(
    extract_from_annotated_document(structured_document, tag_scope=tag_scope)
  )

def parse_args(argv=None):
  parser = argparse.ArgumentParser('Extract JATSy XML from annotated LXML')
  parser.add_argument(
    '--lxml-path', type=str, required=True,
    help='path to lxml or svg pages document'
  )

  parser.add_argument(
    '--tag-scope', type=str, required=False,
    help='tag scope to extract based on'
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

  structured_document = load_structured_document(args.lxml_path)

  xml_root = extract_structured_document_to_xml(
    structured_document,
    tag_scope=args.tag_scope
  )

  get_logger().info('writing result to: %s', args.output_path)
  save_file_content(args.output_path, etree.tostring(xml_root, pretty_print=True))

if __name__ == '__main__':
  logging.basicConfig(level='INFO')

  main()

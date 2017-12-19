import os

from backports.tempfile import TemporaryDirectory

from lxml import etree
from lxml.builder import E

from sciencebeam_gym.utils.xml import (
  get_text_content
)

from sciencebeam_gym.inference_model.extract_from_annotated_document import (
  ExtractedItem
)

from sciencebeam_gym.inference_model.extract_to_xml import (
  extracted_items_to_xml,
  Tags,
  XmlPaths,
  main
)

TEXT_1 = 'some text here'

class TestExtractedItemsToXml(object):
  def test_should_return_empty_xml_for_no_empty_list_of_extracted_items(self):
    xml_root = extracted_items_to_xml([])
    assert xml_root is not None

  def test_should_populate_title(self):
    xml_root = extracted_items_to_xml([
      ExtractedItem(Tags.TITLE, TEXT_1)
    ])
    assert xml_root is not None
    assert get_text_content(xml_root.find(XmlPaths.TITLE)) == TEXT_1

class TestMain(object):
  def test_should_extract_from_simple_annotated_document(self):
    with TemporaryDirectory() as path:
      lxml_root = E.DOCUMENT(
        E.PAGE(
          E.TEXT(
            E.TOKEN(
              TEXT_1,
              {
                'tag': Tags.TITLE
              }
            )
          )
        )
      )

      lxml_path = os.path.join(path, 'test.lxml')
      with open(lxml_path, 'w') as f:
        f.write(etree.tostring(lxml_root))

      output_path = os.path.join(path, 'test.xml')

      main(['--lxml-path=%s' % lxml_path, '--output-path=%s' % output_path])

      xml_root = etree.parse(output_path)
      assert get_text_content(xml_root.find(XmlPaths.TITLE)) == TEXT_1

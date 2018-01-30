import os

from backports.tempfile import TemporaryDirectory

from lxml import etree
from lxml.builder import E

from sciencebeam_gym.utils.xml import (
  get_text_content,
  get_text_content_list
)

from sciencebeam_gym.inference_model.extract_from_annotated_document import (
  ExtractedItem
)

from sciencebeam_gym.inference_model.extract_to_xml import (
  extracted_items_to_xml,
  Tags,
  XmlPaths,
  SubTags,
  SubXmlPaths,
  main
)

TEXT_1 = 'some text here'
TEXT_2 = 'more text to come'
TEXT_3 = 'does not stop here'

def _create_author_extracted_items(given_names, surname):
  return [
    ExtractedItem(Tags.AUTHOR, ' '.join([given_names, surname]), sub_items=[
      ExtractedItem(SubTags.AUTHOR_GIVEN_NAMES, given_names),
      ExtractedItem(SubTags.AUTHOR_SURNAME, surname)
    ])
  ]

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

  def test_should_append_to_abstract(self):
    xml_root = extracted_items_to_xml([
      ExtractedItem(Tags.ABSTRACT, TEXT_1),
      ExtractedItem(Tags.ABSTRACT, TEXT_2)
    ])
    assert xml_root is not None
    assert get_text_content(xml_root.find(XmlPaths.ABSTRACT)) == '\n'.join([TEXT_1, TEXT_2])

  def test_should_not_append_to_abstract_after_untagged_content(self):
    xml_root = extracted_items_to_xml([
      ExtractedItem(Tags.ABSTRACT, TEXT_1),
      ExtractedItem(None, TEXT_2),
      ExtractedItem(Tags.ABSTRACT, TEXT_3)
    ])
    assert xml_root is not None
    assert get_text_content(xml_root.find(XmlPaths.ABSTRACT)) == '\n'.join([TEXT_1, TEXT_3])

  def test_should_not_append_to_abstract_after_another_tag_occured(self):
    xml_root = extracted_items_to_xml([
      ExtractedItem(Tags.ABSTRACT, TEXT_1),
      ExtractedItem(Tags.AUTHOR, TEXT_2),
      ExtractedItem(Tags.ABSTRACT, TEXT_3)
    ])
    assert xml_root is not None
    assert get_text_content(xml_root.find(XmlPaths.ABSTRACT)) == '\n'.join([TEXT_1])

  def test_should_create_separate_author_node(self):
    xml_root = extracted_items_to_xml([
      ExtractedItem(Tags.AUTHOR, TEXT_1),
      ExtractedItem(Tags.AUTHOR, TEXT_2)
    ])
    assert xml_root is not None
    assert get_text_content_list(xml_root.findall(XmlPaths.AUTHOR)) == [TEXT_1, TEXT_2]

  def test_should_extract_author_surname_and_given_names_from_single_author(self):
    xml_root = extracted_items_to_xml([
      ExtractedItem(Tags.AUTHOR, TEXT_1, sub_items=[
        ExtractedItem(SubTags.AUTHOR_GIVEN_NAMES, TEXT_2),
        ExtractedItem(SubTags.AUTHOR_SURNAME, TEXT_3)
      ])
    ])
    assert xml_root is not None
    author = xml_root.find(XmlPaths.AUTHOR)
    assert author is not None
    assert get_text_content(author.find(SubXmlPaths.AUTHOR_GIVEN_NAMES)) == TEXT_2
    assert get_text_content(author.find(SubXmlPaths.AUTHOR_SURNAME)) == TEXT_3

  def test_should_add_contrib_type_author_attribute(self):
    xml_root = extracted_items_to_xml(_create_author_extracted_items(TEXT_1, TEXT_2))
    assert xml_root is not None
    author = xml_root.find(XmlPaths.AUTHOR)
    assert author is not None
    assert author.attrib.get('contrib-type') == 'author'

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

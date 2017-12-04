from lxml.builder import E

from sciencebeam_gym.utils.xml import (
  get_text_content,
  get_immediate_text,
  xml_from_string_with_recover
)

SOME_VALUE_1 = 'some value1'
SOME_VALUE_2 = 'some value2'

class TestGetTextContent(object):
  def test_should_return_simple_text(self):
    node = E.parent(SOME_VALUE_1)
    assert get_text_content(node) == SOME_VALUE_1

  def test_should_return_text_of_child_element(self):
    node = E.parent(E.child(SOME_VALUE_1))
    assert get_text_content(node) == SOME_VALUE_1

  def test_should_return_text_of_child_element_and_preceeding_text(self):
    node = E.parent(SOME_VALUE_1, E.child(SOME_VALUE_2))
    assert get_text_content(node) == SOME_VALUE_1 + SOME_VALUE_2

  def test_should_return_text_of_child_element_and_trailing_text(self):
    node = E.parent(E.child(SOME_VALUE_1), SOME_VALUE_2)
    assert get_text_content(node) == SOME_VALUE_1 + SOME_VALUE_2

  def test_should_return_text_of_parent_excluding_children_to_exclude(self):
    child = E.child(SOME_VALUE_1)
    node = E.parent(child, SOME_VALUE_2)
    assert get_text_content(node, exclude=[child]) == SOME_VALUE_2

class TestGetImmediateText(object):
  def test_should_return_simple_text(self):
    node = E.parent(SOME_VALUE_1)
    assert get_immediate_text(node) == [SOME_VALUE_1]

  def test_should_not_return_text_of_child_element(self):
    node = E.parent(E.child(SOME_VALUE_1))
    assert get_immediate_text(node) == []

class TestXmlFromStringWithRecover(object):
  def test_should_parse_clean_xml(self):
    root = xml_from_string_with_recover('<root><child1>%s</child1></root>' % SOME_VALUE_1)
    node = root.find('child1')
    assert node is not None
    assert node.text == SOME_VALUE_1

  def test_should_parse_xml_with_unencoded_ampersand(self):
    value = 'A & B'
    root = xml_from_string_with_recover('<root><child1>%s</child1></root>' % value)
    node = root.find('child1')
    assert node is not None
    assert node.text == 'A  B'

  def test_should_parse_xml_with_unencoded_unknown_entity(self):
    value = 'A &unknown; B'
    root = xml_from_string_with_recover('<root><child1>%s</child1></root>' % value)
    node = root.find('child1')
    assert node is not None
    assert node.text == 'A  B'

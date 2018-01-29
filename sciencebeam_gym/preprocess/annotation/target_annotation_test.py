from __future__ import division

import json

from lxml.builder import E

from sciencebeam_gym.preprocess.annotation.target_annotation import (
  strip_whitespace,
  xml_root_to_target_annotations,
  XmlMappingSuffix
)

TAG1 = 'tag1'
TAG2 = 'tag2'

SOME_VALUE = 'some value'
SOME_VALUE_2 = 'some value2'
SOME_LONGER_VALUE = 'some longer value1'
SOME_SHORTER_VALUE = 'value1'

class TestStripWhitespace(object):
  def test_should_replace_tab_with_space(self):
    assert strip_whitespace(SOME_VALUE + '\t' + SOME_VALUE_2) == SOME_VALUE + ' ' + SOME_VALUE_2

  def test_should_strip_double_space(self):
    assert strip_whitespace(SOME_VALUE + '  ' + SOME_VALUE_2) == SOME_VALUE + ' ' + SOME_VALUE_2

  def test_should_strip_double_line_feed(self):
    assert strip_whitespace(SOME_VALUE + '\n\n' + SOME_VALUE_2) == SOME_VALUE + '\n' + SOME_VALUE_2

  def test_should_replace_cr_with_line_feed(self):
    assert strip_whitespace(SOME_VALUE + '\r' + SOME_VALUE_2) == SOME_VALUE + '\n' + SOME_VALUE_2

  def test_should_strip_spaces_around_line_feed(self):
    assert strip_whitespace(SOME_VALUE + ' \n ' + SOME_VALUE_2) == SOME_VALUE + '\n' + SOME_VALUE_2

  def test_should_strip_multiple_lines_with_blanks(self):
    assert (
      strip_whitespace(SOME_VALUE + ' \n \n \n ' + SOME_VALUE_2) ==
      SOME_VALUE + '\n' + SOME_VALUE_2
    )

class TestXmlRootToTargetAnnotations(object):
  def test_should_return_empty_target_annotations_for_empty_xml(self):
    xml_root = E.article(
    )
    xml_mapping = {
      'article': {
        'title': 'title'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert target_annotations == []

  def test_should_return_empty_target_annotations_for_no_matching_annotations(self):
    xml_root = E.article(
      E.other(SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert target_annotations == []

  def test_should_return_matching_target_annotations(self):
    xml_root = E.article(
      E.title(SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert len(target_annotations) == 1
    assert target_annotations[0].name == TAG1
    assert target_annotations[0].value == SOME_VALUE

  def test_should_strip_extra_space(self):
    xml_root = E.article(
      E.abstract(SOME_VALUE + '  ' + SOME_VALUE_2)
    )
    xml_mapping = {
      'article': {
        TAG1: 'abstract'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert len(target_annotations) == 1
    assert target_annotations[0].name == TAG1
    assert target_annotations[0].value == SOME_VALUE + ' ' + SOME_VALUE_2

  def test_should_apply_regex_to_result(self):
    xml_root = E.article(
      E.title('1.1. ' + SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title',
        TAG1 + XmlMappingSuffix.REGEX: r'(?:\d+\.?)* ?(.*)'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert len(target_annotations) == 1
    assert target_annotations[0].name == TAG1
    assert target_annotations[0].value == SOME_VALUE

  def test_should_apply_match_multiple_flag(self):
    xml_root = E.article(
      E.title(SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title',
        TAG1 + XmlMappingSuffix.MATCH_MULTIPLE: 'true'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [t.match_multiple for t in target_annotations] == [True]

  def test_should_not_apply_match_multiple_flag_if_not_set(self):
    xml_root = E.article(
      E.title(SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [t.match_multiple for t in target_annotations] == [False]

  def test_should_apply_match_bonding_flag(self):
    xml_root = E.article(
      E.title(SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title',
        TAG1 + XmlMappingSuffix.BONDING: 'true'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [t.bonding for t in target_annotations] == [True]

  def test_should_not_apply_match_bonding_flag_if_not_set(self):
    xml_root = E.article(
      E.title(SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [t.bonding for t in target_annotations] == [False]

  def test_should_apply_match_require_next_flag(self):
    xml_root = E.article(
      E.title(SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title',
        TAG1 + XmlMappingSuffix.REQUIRE_NEXT: 'true'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [t.require_next for t in target_annotations] == [True]

  def test_should_not_apply_match_require_next_flag_if_not_set(self):
    xml_root = E.article(
      E.title(SOME_VALUE)
    )
    xml_mapping = {
      'article': {
        TAG1: 'title'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [t.require_next for t in target_annotations] == [False]

  def test_should_use_multiple_xpaths(self):
    xml_root = E.article(
      E.entry(
        E.child1(SOME_VALUE),
        E.child2(SOME_VALUE_2)
      )
    )
    xml_mapping = {
      'article': {
        TAG1: '\n{}\n{}\n'.format(
          'entry/child1',
          'entry/child2'
        )
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, SOME_VALUE),
      (TAG1, SOME_VALUE_2)
    ]

  def test_should_apply_children_xpaths_and_sort_by_value_descending(self):
    xml_root = E.article(
      E.entry(
        E.child1(SOME_SHORTER_VALUE),
        E.child2(SOME_LONGER_VALUE)
      ),
      E.entry(
        E.child1(SOME_LONGER_VALUE)
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: './/*'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, [SOME_LONGER_VALUE, SOME_SHORTER_VALUE]),
      (TAG1, SOME_LONGER_VALUE)
    ]

  def test_should_apply_children_xpaths_and_exclude_parents(self):
    xml_root = E.article(
      E.entry(
        E.parent(
          E.child2(SOME_LONGER_VALUE),
          E.child1(SOME_SHORTER_VALUE)
        )
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: './/*'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, [SOME_LONGER_VALUE, SOME_SHORTER_VALUE])
    ]

  def test_should_apply_children_xpaths_and_include_parent_text_between_matched_children(self):
    xml_root = E.article(
      E.entry(
        E.parent(
          E.child2(SOME_LONGER_VALUE),
          SOME_VALUE,
          E.child1(SOME_SHORTER_VALUE)
        )
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: './/*'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, [SOME_LONGER_VALUE, SOME_VALUE, SOME_SHORTER_VALUE])
    ]

  def test_should_apply_multiple_children_xpaths_and_include_parent_text_if_enabled(self):
    xml_root = E.article(
      E.entry(
        E.child1(SOME_SHORTER_VALUE),
        SOME_LONGER_VALUE
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: '\n{}\n{}\n'.format('.//*', '.'),
        TAG1 + XmlMappingSuffix.UNMATCHED_PARENT_TEXT: 'true'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, [SOME_LONGER_VALUE, SOME_SHORTER_VALUE])
    ]

  def test_should_apply_concat_children(self):
    num_values = ['101', '202']
    xml_root = E.article(
      E.entry(
        E.parent(
          E.child1(SOME_VALUE),
          E.fpage(num_values[0]),
          E.lpage(num_values[1])
        )
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: './/*',
        TAG1 + XmlMappingSuffix.CHILDREN_CONCAT: json.dumps([[{
          'xpath': './/fpage'
        }, {
          'value': '-'
        }, {
          'xpath': './/lpage'
        }]])
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, [SOME_VALUE, '-'.join(num_values)])
    ]

  def test_should_not_apply_concat_children_if_one_node_was_not_found(self):
    num_values = ['101', '202']
    xml_root = E.article(
      E.entry(
        E.parent(
          E.child1(SOME_VALUE),
          E.fpage(num_values[0]),
          E.lpage(num_values[1])
        )
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: './/*',
        TAG1 + XmlMappingSuffix.CHILDREN_CONCAT: json.dumps([[{
          'xpath': './/fpage'
        }, {
          'value': '-'
        }, {
          'xpath': './/unknown'
        }]])
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, [SOME_VALUE, num_values[0], num_values[1]])
    ]

  def test_should_apply_range_children(self):
    num_values = [101, 102, 103, 104, 105, 106, 107]
    xml_root = E.article(
      E.entry(
        E.child1(SOME_VALUE),
        E.fpage(str(min(num_values))),
        E.lpage(str(max(num_values)))
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: 'fpage|lpage',
        TAG1 + XmlMappingSuffix.CHILDREN_RANGE: json.dumps([{
          'min': {
            'xpath': 'fpage'
          },
          'max': {
            'xpath': 'lpage'
          }
        }])
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, [str(x) for x in num_values])
    ]

  def test_should_apply_range_children_as_separate_target_annotations(self):
    num_values = [101, 102, 103, 104, 105, 106, 107]
    xml_root = E.article(
      E.entry(
        E.child1(SOME_VALUE),
        E.fpage(str(min(num_values))),
        E.lpage(str(max(num_values)))
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: 'fpage|lpage',
        TAG1 + XmlMappingSuffix.CHILDREN_RANGE: json.dumps([{
          'min': {
            'xpath': 'fpage'
          },
          'max': {
            'xpath': 'lpage'
          },
          'standalone': True
        }])
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, str(x))
      for x in num_values
    ]

  def test_should_not_apply_range_children_if_xpath_not_matching(self):
    num_values = [101, 102, 103, 104, 105, 106, 107]
    fpage = str(min(num_values))
    lpage = str(max(num_values))
    xml_root = E.article(
      E.entry(
        E.child1(SOME_VALUE),
        E.fpage(fpage),
        E.lpage(lpage)
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: 'fpage|unknown',
        TAG1 + XmlMappingSuffix.CHILDREN_RANGE: json.dumps([{
          'min': {
            'xpath': 'fpage'
          },
          'max': {
            'xpath': 'unknown'
          }
        }])
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, fpage)
    ]

  def test_should_not_apply_range_children_if_value_is_not_integer(self):
    fpage = 'abc'
    lpage = 'xyz'
    xml_root = E.article(
      E.entry(
        E.child1(SOME_VALUE),
        E.fpage(fpage),
        E.lpage(lpage)
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.CHILDREN: 'fpage|lpage',
        TAG1 + XmlMappingSuffix.CHILDREN_RANGE: json.dumps([{
          'min': {
            'xpath': 'fpage'
          },
          'max': {
            'xpath': 'lpage'
          }
        }])
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations] == [
      (TAG1, [fpage, lpage])
    ]

  def test_should_add_sub_annotations(self):
    xml_root = E.article(
      E.entry(
        E.firstname(SOME_VALUE),
        E.givennames(SOME_VALUE_2)
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.SUB + '.firstname': './firstname',
        TAG1 + XmlMappingSuffix.SUB + '.givennames': './givennames',
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations[0].sub_annotations] == [
      ('firstname', SOME_VALUE),
      ('givennames', SOME_VALUE_2)
    ]

  def test_should_add_sub_annotations_with_multiple_values(self):
    xml_root = E.article(
      E.entry(
        E.value(SOME_VALUE),
        E.value(SOME_VALUE_2)
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'entry',
        TAG1 + XmlMappingSuffix.SUB + '.value': './value'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(t.name, t.value) for t in target_annotations[0].sub_annotations] == [
      ('value', SOME_VALUE),
      ('value', SOME_VALUE_2)
    ]

  def test_should_return_full_text(self):
    xml_root = E.article(
      E.title(
        'some ',
        E.other('embedded'),
        ' text'
      )
    )
    xml_mapping = {
      'article': {
        TAG1: 'title'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert len(target_annotations) == 1
    assert target_annotations[0].name == TAG1
    assert target_annotations[0].value == 'some embedded text'

  def test_should_return_target_annotations_in_order_of_xml(self):
    xml_root = E.article(
      E.tag1('tag1.1'), E.tag2('tag2.1'), E.tag1('tag1.2'), E.tag2('tag2.2'),
    )
    xml_mapping = {
      'article': {
        TAG1: 'tag1',
        TAG2: 'tag2'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(ta.name, ta.value) for ta in target_annotations] == [
      (TAG1, 'tag1.1'), (TAG2, 'tag2.1'), (TAG1, 'tag1.2'), (TAG2, 'tag2.2')
    ]

  def test_should_return_target_annotations_in_order_of_priority_first(self):
    xml_root = E.article(
      E.tag1('tag1.1'), E.tag2('tag2.1'), E.tag1('tag1.2'), E.tag2('tag2.2'),
    )
    xml_mapping = {
      'article': {
        TAG1: 'tag1',
        TAG2: 'tag2',
        TAG2 + XmlMappingSuffix.PRIORITY: '1'
      }
    }
    target_annotations = xml_root_to_target_annotations(xml_root, xml_mapping)
    assert [(ta.name, ta.value) for ta in target_annotations] == [
      (TAG2, 'tag2.1'), (TAG2, 'tag2.2'), (TAG1, 'tag1.1'), (TAG1, 'tag1.2')
    ]

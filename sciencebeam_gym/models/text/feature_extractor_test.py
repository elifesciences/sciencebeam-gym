from sciencebeam_gym.utils.bounding_box import (
  BoundingBox
)

from sciencebeam_gym.structured_document import (
  SimpleStructuredDocument,
  SimplePage,
  SimpleLine,
  SimpleToken
)

from sciencebeam_gym.models.text.feature_extractor import (
  structured_document_to_token_props,
  token_props_list_to_features,
  token_props_list_to_labels,
  remove_labels_from_token_props_list,
  NONE_TAG
)

PAGE_BOUNDING_BOX = BoundingBox(0, 0, 100, 200)
TOKEN_BOUNDING_BOX = BoundingBox(10, 10, 10, 20)

TEXT_1 = 'Text 1'
TEXT_2 = 'Text 2'
TEXT_3 = 'Text 3'

TAG_1 = 'tag1'
TAG_2 = 'tag2'
TAG_3 = 'tag3'

class TestStructuredDocumentToTokenProps(object):
  def test_should_return_empty_token_list_if_document_has_no_pages(self):
    structured_document = SimpleStructuredDocument([])
    assert list(structured_document_to_token_props(
      structured_document
    )) == []

  def test_should_return_empty_token_list_if_document_has_no_lines(self):
    structured_document = SimpleStructuredDocument(
      SimplePage([], bounding_box=PAGE_BOUNDING_BOX)
    )
    assert list(structured_document_to_token_props(
      structured_document
    )) == []

  def test_should_return_single_token_text(self):
    structured_document = SimpleStructuredDocument(
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    )
    result = list(structured_document_to_token_props(
      structured_document
    ))
    assert [t.get('text') for t in result] == [TEXT_1]

  def test_should_return_multiple_token_texts(self):
    structured_document = SimpleStructuredDocument(
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1),
        SimpleToken(TEXT_2),
        SimpleToken(TEXT_3)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    )
    result = list(structured_document_to_token_props(
      structured_document
    ))
    assert [t.get('text') for t in result] == [TEXT_1, TEXT_2, TEXT_3]

  def test_should_return_tag(self):
    structured_document = SimpleStructuredDocument([
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1, tag=TAG_1),
        SimpleToken(TEXT_2)
      ])], bounding_box=PAGE_BOUNDING_BOX),
      SimplePage([SimpleLine([
        SimpleToken(TEXT_3, tag=TAG_3)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    ])
    result = list(structured_document_to_token_props(
      structured_document
    ))
    assert [t.get('tag') for t in result] == [TAG_1, None, TAG_3]

  def test_should_return_bounding_box(self):
    structured_document = SimpleStructuredDocument([
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1, bounding_box=TOKEN_BOUNDING_BOX)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    ])
    result = list(structured_document_to_token_props(
      structured_document
    ))
    assert [t.get('bounding_box') for t in result] == [TOKEN_BOUNDING_BOX]

  def test_should_return_rel_bounding_box(self):
    structured_document = SimpleStructuredDocument([
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1, bounding_box=TOKEN_BOUNDING_BOX)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    ])
    result = list(structured_document_to_token_props(
      structured_document
    ))
    assert [t.get('rel_bounding_box') for t in result] == [
      TOKEN_BOUNDING_BOX.scale_by(
        1.0 / PAGE_BOUNDING_BOX.width,
        1.0 / PAGE_BOUNDING_BOX.height
      )
    ]

  def test_should_return_page_index_and_page_count(self):
    structured_document = SimpleStructuredDocument([
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1),
        SimpleToken(TEXT_2)
      ])], bounding_box=PAGE_BOUNDING_BOX),
      SimplePage([SimpleLine([
        SimpleToken(TEXT_3)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    ])
    result = list(structured_document_to_token_props(
      structured_document
    ))
    pages = [t.get('page') for t in result]
    assert [p.get('index') for p in pages] == [0, 0, 1]
    assert [p.get('count') for p in pages] == [2, 2, 2]

  def test_should_return_page_width_and_height(self):
    structured_document = SimpleStructuredDocument([
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    ])
    result = list(structured_document_to_token_props(
      structured_document
    ))
    pages = [t.get('page') for t in result]
    assert [p.get('width') for p in pages] == [PAGE_BOUNDING_BOX.width]
    assert [p.get('height') for p in pages] == [PAGE_BOUNDING_BOX.height]

  def test_should_return_line_index_and_page_count(self):
    structured_document = SimpleStructuredDocument([
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1)
      ]), SimpleLine([
        SimpleToken(TEXT_2)
      ])], bounding_box=PAGE_BOUNDING_BOX),
      SimplePage([SimpleLine([
        SimpleToken(TEXT_3)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    ])
    result = list(structured_document_to_token_props(
      structured_document
    ))
    lines = [t.get('line') for t in result]
    assert [l.get('index') for l in lines] == [0, 1, 0]
    assert [l.get('count') for l in lines] == [2, 2, 1]

  def test_should_return_line_token_index_and_page_count(self):
    structured_document = SimpleStructuredDocument([
      SimplePage([SimpleLine([
        SimpleToken(TEXT_1),
        SimpleToken(TEXT_2)
      ])], bounding_box=PAGE_BOUNDING_BOX),
      SimplePage([SimpleLine([
        SimpleToken(TEXT_3)
      ])], bounding_box=PAGE_BOUNDING_BOX)
    ])
    result = list(structured_document_to_token_props(
      structured_document
    ))
    line_tokens = [t.get('line_token') for t in result]
    assert [t.get('index') for t in line_tokens] == [0, 1, 0]
    assert [t.get('count') for t in line_tokens] == [2, 2, 1]

def create_token_props(text, **kwargs):
  d = {
    'text': text
  }
  d.update(kwargs)
  return d

class TestTokenPropsListToFeatures(object):
  def test_should_extract_various_word_features(self):
    result = token_props_list_to_features([
      create_token_props('TestMe')
    ])
    assert [x.get('word.lower') for x in result] == ['testme']
    assert [x.get('word[:1]') for x in result] == ['t']
    assert [x.get('word[-3:]') for x in result] == ['tme']
    assert [x.get('word[-2:]') for x in result] == ['me']
    assert [x.get('word[:1].isupper') for x in result] == [True]
    assert [x.get('word.isupper') for x in result] == [False]
    assert [x.get('word.isdigit') for x in result] == [False]

  def test_should_add_previous_and_next_token_word_features(self):
    result = token_props_list_to_features([
      create_token_props(TEXT_1),
      create_token_props(TEXT_2),
      create_token_props(TEXT_3)
    ])
    assert [x.get('word.lower') for x in result] == [
      TEXT_1.lower(), TEXT_2.lower(), TEXT_3.lower()
    ]
    assert [x.get('-2:word.lower') for x in result] == [
      None, None, TEXT_1.lower()
    ]
    assert [x.get('-1:word.lower') for x in result] == [
      None, TEXT_1.lower(), TEXT_2.lower()
    ]
    assert [x.get('1:word.lower') for x in result] == [
      TEXT_2.lower(), TEXT_3.lower(), None
    ]
    assert [x.get('2:word.lower') for x in result] == [
      TEXT_3.lower(), None, None
    ]
    assert [x.get('BOD[-2]') for x in result] == [
      True, True, None
    ]
    assert [x.get('BOD[-1]') for x in result] == [
      True, None, None
    ]
    assert [x.get('EOD[1]') for x in result] == [
      None, None, True
    ]
    assert [x.get('EOD[2]') for x in result] == [
      None, True, True
    ]

  def test_should_not_include_tag(self):
    result = token_props_list_to_features([
      create_token_props(TEXT_1, tag=TAG_1)
    ])
    assert [x.get('tag') for x in result] == [None]

class TestTokenPropsListToLabels(object):
  def test_should_extract_tag(self):
    assert token_props_list_to_labels([
      create_token_props(TEXT_1, tag=TAG_1),
      create_token_props(TEXT_2, tag=TAG_2)
    ]) == [TAG_1, TAG_2]

  def test_should_replace_none_tag(self):
    assert token_props_list_to_labels([
      create_token_props(TEXT_1, tag=TAG_1),
      create_token_props(TEXT_2, tag=None)
    ]) == [TAG_1, NONE_TAG]

class TestRemoveLabelsFromTokenPropsList(object):
  def test_should_remove_tag(self):
    token_props_list = [
      create_token_props(TEXT_1, tag=TAG_1),
      create_token_props(TEXT_2, tag=TAG_2)
    ]
    updated_token_props_list = remove_labels_from_token_props_list(token_props_list)
    assert [x.get('tag') for x in token_props_list] == [TAG_1, TAG_2]
    assert [x.get('tag') for x in updated_token_props_list] == [None, None]
    assert [x.get('text') for x in updated_token_props_list] == [TEXT_1, TEXT_2]

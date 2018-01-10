
NONE_TAG = 'O'

def structured_document_to_token_props(structured_document):
  pages = list(structured_document.get_pages())
  for page_index, page in enumerate(pages):
    page_bounding_box = structured_document.get_bounding_box(page)
    assert page_bounding_box is not None
    page_width = page_bounding_box.width
    page_height = page_bounding_box.height
    page_info = {
      'index': page_index,
      'count': len(pages),
      'width': page_width,
      'height': page_height
    }
    page_rx = 1.0 / page_width
    page_ry = 1.0 / page_height
    lines = list(structured_document.get_lines_of_page(page))
    for line_index, line in enumerate(lines):
      line_tokens = list(structured_document.get_tokens_of_line(line))
      line_info = {
        'index': line_index,
        'count': len(lines)
      }
      for line_token_index, token in enumerate(line_tokens):
        line_token_info = {
          'index': line_token_index,
          'count': len(line_tokens)
        }
        bounding_box = structured_document.get_bounding_box(token)
        rel_bounding_box = (
          bounding_box.scale_by(page_rx, page_ry)
          if bounding_box else None
        )
        yield {
          'text': structured_document.get_text(token),
          'tag': structured_document.get_tag(token),
          'scoped_tags': {
            k: v
            for k, v in structured_document.get_tag_by_scope(token).items()
            if k
          },
          'bounding_box': bounding_box,
          'rel_bounding_box': rel_bounding_box,
          'line_token': line_token_info,
          'page': page_info,
          'line': line_info
        }

def token_props_features(token_props, prefix=''):
  word = token_props.get('text') or ''
  word_lower = word.lower()
  d = {
    prefix + 'word.lower': word_lower,
    prefix + 'word[:1]': word_lower[:1],
    prefix + 'word[-3:]': word_lower[-3:],
    prefix + 'word[-2:]': word_lower[-2:],
    prefix + 'word[:1].isupper': word[:1].istitle(),
    prefix + 'word.isupper': word.isupper(),
    prefix + 'word.isdigit': word.isdigit()
  }
  for scope, tag in token_props.get('scoped_tags', {}).items():
    d[prefix + scope + '.tag'] = tag
  return d

def token_props_to_features(token_props_list, i):
  features = token_props_features(token_props_list[i])
  if i > 0:
    pass
  for rel_token_index in [-2, -1, 1, 2]:
    abs_token_index = i + rel_token_index
    if abs_token_index < 0:
      features['BOD[%d]' % rel_token_index] = True
    elif abs_token_index >= len(token_props_list):
      features['EOD[%d]' % rel_token_index] = True
    else:
      features.update(token_props_features(
        token_props_list[abs_token_index],
        str(rel_token_index) + ':'
      ))
  return features


def token_props_list_to_features(token_props_list):
  token_props_list = list(token_props_list)
  return [token_props_to_features(token_props_list, i) for i in range(len(token_props_list))]

def remove_labels_from_token_props_list(token_props_list):
  return [
    {k: v for k, v in token_props.items() if k != 'tag'}
    for token_props in token_props_list
  ]

def token_props_list_to_labels(token_props_list):
  return [token_props.get('tag') or NONE_TAG for token_props in token_props_list]

class ExtractedItem(object):
  def __init__(self, tag, text):
    self.tag = tag
    self.text = text

  def extend(self, other_item):
    return ExtractedItem(
      self.tag,
      self.text + '\n' + other_item.text
    )

def get_lines(structured_document):
  for page in structured_document.get_pages():
    for line in structured_document.get_lines_of_page(page):
      yield line

def extract_from_annotated_tokens(structured_document, tokens):
  previous_tokens = []
  previous_tag = None
  for token in tokens:
    tag = structured_document.get_tag(token)
    if not previous_tokens:
      previous_tokens = [token]
      previous_tag = tag
    elif tag == previous_tag:
      previous_tokens.append(token)
    else:
      yield ExtractedItem(
        previous_tag,
        ' '.join(structured_document.get_text(t) for t in previous_tokens)
      )
      previous_tokens = [token]
      previous_tag = tag
  if previous_tokens:
    yield ExtractedItem(
      previous_tag,
      ' '.join(structured_document.get_text(t) for t in previous_tokens)
    )

def extract_from_annotated_lines(structured_document, lines):
  previous_item = None
  for line in lines:
    tokens = structured_document.get_tokens_of_line(line)
    for item in extract_from_annotated_tokens(structured_document, tokens):
      if previous_item is not None:
        if previous_item.tag == item.tag:
          previous_item = previous_item.extend(item)
        else:
          yield previous_item
          previous_item = item
      else:
        previous_item = item
  if previous_item is not None:
    yield previous_item

def extract_from_annotated_document(structured_document):
  for x in extract_from_annotated_lines(structured_document, get_lines(structured_document)):
    yield x

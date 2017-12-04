from lxml import etree

def _get_text_content_and_exclude(node, exclude):
  result = ''
  if node.text is not None:
    result += node.text
  result += ''.join([
    (
      _get_text_content_and_exclude(c, exclude)
      if c not in exclude
      else ''
    ) +
    (c.tail if c.tail is not None else '')
    for c in node.iterchildren()
  ])
  return result

def get_text_content(node, exclude=None):
  '''
  Strip tags and return text content
  '''
  if not exclude:
    return ''.join(node.itertext())
  return _get_text_content_and_exclude(node, exclude)

def get_immediate_text(node):
  return node.xpath('text()')

def get_text_content_list(nodes, exclude=None):
  return [get_text_content(node, exclude=exclude) for node in nodes]

def xml_from_string_with_recover(s):
  parser = etree.XMLParser(recover=True)
  return etree.fromstring(s, parser=parser)

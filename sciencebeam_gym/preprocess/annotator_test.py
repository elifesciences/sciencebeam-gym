from sciencebeam_gym.preprocess.annotator import (
  LineAnnotator
)

from sciencebeam_gym.structured_document import (
  SimpleLine,
  SimpleToken,
  SimpleStructuredDocument
)

line_annotator = LineAnnotator()

class TestLineAnnotator(object):
  def test_x(self):
    line_number_tokens = [
      SimpleToken(str(line_no), dict(x='1', y=str(line_no * 20)))
      for line_no in range(1, 5)
    ]
    doc = SimpleStructuredDocument(lines=[
      SimpleLine([
        line_number_token,
        SimpleToken('other text', dict(
          x=str(float(line_number_token.get_x()) + 50),
          y=line_number_token.get_y()
        ))
      ])
      for line_number_token in line_number_tokens
    ])
    line_annotator.annotate(
      doc
    )
    assert [t.get_tag() for t in line_number_tokens] == ['line_no'] * len(line_number_tokens)

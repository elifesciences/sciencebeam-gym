from __future__ import division

import logging
from collections import Counter

# Max absolute variation of x from the determined x for line numbers
DEFAULT_X_THRESHOLD = 10

# Minimum ratio of tokens within x threshold vs total number tokens on page
# (low ratio indicates numbers may be figures or table values rather than line numbers)
DEFAULT_TOKEN_RATIO_THRESHOLD = 0.7

def get_logger():
  return logging.getLogger(__name__)

def _find_line_number_token_candidates(structured_document, page):
  for line in structured_document.get_lines_of_page(page):
    text_tokens = sorted(
      structured_document.get_tokens_of_line(line),
      key=lambda t: structured_document.get_x(t)
    )
    if text_tokens:
      token = text_tokens[0]
      token_text = structured_document.get_text(token)
      if token_text and token_text.isdigit():
        yield token

def find_line_number_tokens(
  structured_document,
  x_threshold=DEFAULT_X_THRESHOLD,
  token_ratio_threshold=DEFAULT_TOKEN_RATIO_THRESHOLD):

  for page in structured_document.get_pages():
    line_number_candidates = list(_find_line_number_token_candidates(
      structured_document,
      page
    ))
    # we need more than two lines
    if len(line_number_candidates) > 2:
      c = Counter((
        round(float(structured_document.get_x(t)))
        for t in line_number_candidates
      ))
      get_logger().debug('counter: %s', c)
      most_common_x, most_common_count = c.most_common(1)[0]
      get_logger().debug('most_common: x: %s (count: %s)', most_common_x, most_common_count)
      tokens_within_range = [
        token
        for token in line_number_candidates
        if abs(float(token.attrib['x']) - most_common_x) < x_threshold
      ]
      token_within_range_ratio = len(tokens_within_range) / len(line_number_candidates)
      if token_within_range_ratio < token_ratio_threshold:
        get_logger().debug(
          'token within range ratio not meeting threshold: %f < %f',
          token_within_range_ratio,
          token_ratio_threshold
        )
      else:
        for token in tokens_within_range:
          yield token

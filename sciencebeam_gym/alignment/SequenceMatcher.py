import platform
import warnings

try:
  from fuzzywuzzy.StringMatcher import StringMatcher as SequenceMatcher
except ImportError:
  if platform.python_implementation() != "PyPy":
    warnings.warn(
      'Using slow pure-python SequenceMatcher.'
      ' Install python-Levenshtein (and fuzzywuzzy) to remove this warning'
    )
  from difflib import SequenceMatcher

assert SequenceMatcher is not None

from abc import ABCMeta, abstractmethod
from typing import List

from sciencebeam_gym.preprocess.annotation.find_line_number import (
    find_line_number_tokens
)


class AbstractAnnotator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def annotate(self, structured_document):
        pass


class LineAnnotator(AbstractAnnotator):
    def __init__(self, tag='line_no'):
        self.tag = tag

    def annotate(self, structured_document):
        for t in find_line_number_tokens(structured_document):
            structured_document.set_tag(t, self.tag)
        return structured_document


DEFAULT_ANNOTATORS: List[AbstractAnnotator] = [
    LineAnnotator()
]


class Annotator(object):
    def __init__(self, annotators=None):
        if annotators is None:
            annotators = DEFAULT_ANNOTATORS
        self.annotators = annotators

    def annotate(self, structured_document):
        for annotator in self.annotators:
            structured_document = annotator.annotate(structured_document)
        return structured_document

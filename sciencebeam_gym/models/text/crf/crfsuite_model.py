import logging

from sklearn_crfsuite import CRF


def get_logger():
    return logging.getLogger(__name__)


DEFAULT_PARAMS = dict(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)


class CrfSuiteModel(CRF):
    def __init__(self, **kwargs):
        d = dict(DEFAULT_PARAMS)
        d.update(kwargs)
        super(CrfSuiteModel, self).__init__(**d)

import logging

import pytest


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    logging.root.setLevel('WARNING')
    for name in ['sciencebeam_utils', 'sciencebeam_alignment', 'sciencebeam_gym']:
        logging.getLogger(name).setLevel('DEBUG')
    logging.getLogger('tensorflow').setLevel('ERROR')

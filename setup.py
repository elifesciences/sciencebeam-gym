from __future__ import print_function

import os

from setuptools import find_packages
from setuptools import setup

with open(os.path.join('requirements.txt'), 'r') as f:
  REQUIRED_PACKAGES = f.readlines()

packages = find_packages()

setup(
    name='sciencebeam_gym',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    include_package_data=True,
    description='ScienceBeam Gym'
)

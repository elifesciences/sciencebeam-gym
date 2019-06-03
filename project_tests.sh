#!/bin/bash
set -e

echo "running flake8"
flake8 sciencebeam_gym tests setup.py

echo "running pylint"
pylint sciencebeam_gym tests setup.py

echo "running tests"
pytest

echo "done"

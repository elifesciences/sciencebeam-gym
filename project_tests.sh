#!/bin/bash
set -e

echo "running flake8"
flake8 sciencebeam_gym setup.py

echo "running pylint"
pylint sciencebeam_gym setup.py

echo "running tests"
pytest sciencebeam_gym

echo "done"

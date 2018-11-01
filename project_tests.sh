#!/bin/bash
set -e

pip install -r requirements.dev.txt

pytest sciencebeam_gym

echo "running pylint"
pylint sciencebeam_gym setup.py

echo "done"

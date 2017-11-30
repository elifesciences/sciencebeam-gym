#!/bin/bash
set -e

docker run elife/sciencebeam-gym /bin/bash -c 'venv/bin/pip install pytest && venv/bin/pytest sciencebeam_gym/preprocess/find_line_numbers_test.py'

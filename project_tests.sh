#!/bin/bash
set -e

docker run elife/sciencebeam-gym /bin/bash -c 'venv/bin/pip install pytest nose && venv/bin/pytest sciencebeam_gym/preprocess'

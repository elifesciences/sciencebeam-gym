#!/bin/bash
set -e

docker run --rm elife/sciencebeam-gym /bin/bash -c 'venv/bin/pip install pytest nose && venv/bin/pytest sciencebeam_gym'

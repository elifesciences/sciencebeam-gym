#!/bin/bash

set -e

python -m sciencebeam_gym.preprocess.preprocessing_pipeline $@

#!/bin/bash

source prepare-shell.sh

python -m sciencebeam_gym.tools.inspect_tfrecords \
  --records_paths "${TRAIN_PREPROC_PATH}/*tfrecord*" \
  --inspect_key "input_uri" \
  --extract_dir ".temp/train" \
  --extract_image "input_image" \
  --extract_image "annotation_image"

python -m sciencebeam_gym.tools.inspect_tfrecords \
  --records_paths "${EVAL_PREPROC_PATH}/*tfrecord*" \
  --inspect_key "input_uri" \
  --extract_dir ".temp/eval" \
  --extract_image "input_image" \
  --extract_image "annotation_image"

if [ ! -z "$QUALITATIVE_PREPROC_PATH" ]; then
  python -m sciencebeam_gym.tools.inspect_tfrecords \
    --records_paths "${QUALITATIVE_PREPROC_PATH}/*tfrecord*" \
    --inspect_key "input_uri" \
    --extract_dir ".temp/eval" \
    --extract_image "input_image" \
    --extract_image "annotation_image"
fi

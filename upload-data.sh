#!/bin/bash

source prepare-shell.sh

data_dir="$1"

if [ -z "$data_dir" ]; then
  data_dir="${LOCAL_DATA_PATH}"
fi

echo "Data directory: $data_dir"
echo "Target: $GCS_DATA_PATH"

train_data_dir="$data_dir/train"
test_data_dir="$data_dir/test"

if [ ! -d "$train_data_dir" ]; then
  echo "Training data directory missing: $train_data_dir"
  exit 2
fi

if [ ! -d "$test_data_dir" ]; then
  echo "Testing data directory missing: $test_data_dir"
  exit 2
fi

if [ ! -z "$QUANTITATIVE_FOLDER_NAME" ]; then
  quantitative_data_dir="$data_dir/$QUANTITATIVE_FOLDER_NAME"
  if [ ! -d "$quantitative_data_dir" ]; then
    echo "Quantitative data directory missing: $quantitative_data_dir"
    exit 2
  fi
fi

gsutil -m cp -r "$train_data_dir" "$GCS_DATA_PATH/"
gsutil -m cp -r "$test_data_dir" "$GCS_DATA_PATH/"

if [ ! -z "$QUANTITATIVE_FOLDER_NAME" ]; then
  gsutil -m cp -r "$quantitative_data_dir" "$GCS_DATA_PATH/"
fi

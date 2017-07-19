#!/bin/bash

export JOB_ID_FILE='.job-id'
if [ -f "$JOB_ID_FILE" ]; then
  rm "${JOB_ID_FILE}"
fi

source prepare-shell.sh

COMMON_ARGS=(
  --output_path "${TRAIN_MODEL_PATH}/"
  --eval_data_paths "${PREPROC_PATH}/test/*tfrecord*"
  --train_data_paths "${PREPROC_PATH}/train/*tfrecord*"
  --model "${MODEL_NAME}"
  --batch_size 10
  --eval_set_size 10
)

if [ $USE_SEPARATE_CHANNELS == true ]; then
  COMMON_ARGS=(
    ${COMMON_ARGS[@]}
    --color_map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}"
  )
fi

if [ $USE_CLOUD == true ]; then
  gcloud ml-engine jobs submit training "$JOB_ID" \
    --stream-logs \
    --module-name sciencebeam_gym.trainer.task \
    --package-path sciencebeam_gym \
    --staging-bucket "$BUCKET" \
    --region us-central1 \
    --runtime-version=1.0 \
    --scale-tier=BASIC_GPU \
    -- \
    --save_max_to_keep 10 \
    --log_interval_secs 100000 \
    --eval_interval_secs 100000 \
    --save_interval_secs 100000 \
    --log_freq 500 \
    --eval_freq 500 \
    --save_freq 500 \
    --max_steps 1000 \
    ${COMMON_ARGS[@]}
else
  gcloud ml-engine local train \
    --module-name sciencebeam_gym.trainer.task \
    --package-path sciencebeam_gym.trainer \
    -- \
    --save_max_to_keep 3 \
    --log_interval_secs 600 \
    --eval_interval_secs 300 \
    --save_interval_secs 300 \
    --log_freq 50 \
    --eval_freq 50 \
    --save_freq 50 \
    --max_steps 3 \
    ${COMMON_ARGS[@]}
fi

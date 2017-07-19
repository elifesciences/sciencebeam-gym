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
  --max_steps 0
  --write_predictions
  ${TRAINING_ARGS}
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
    -- \
    --cloud \
    ${COMMON_ARGS[@]}
else
  gcloud ml-engine local train \
    --module-name sciencebeam_gym.trainer.task \
    --package-path sciencebeam_gym.trainer \
    -- \
    ${COMMON_ARGS[@]}
fi

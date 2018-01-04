#!/bin/bash

set -e

source prepare-shell.sh

PREDICT_INPUT="$1"
PREDICT_OUTPUT="$2"

if [ -z "$PREDICT_INPUT" ]; then
  echo "Usage: $0 <predict input> [<predict output>]"
  exit 1
fi

if [ -z "$PREDICT_OUTPUT" ]; then
  PREDICT_OUTPUT=$PREDICT_INPUT.out.png
fi

COMMON_ARGS=(
  --output_path "${TRAIN_MODEL_PATH}/"
  --train_data_paths "${TRAIN_PREPROC_PATH}/*tfrecord*"
  --eval_data_paths "${EVAL_PREPROC_PATH}/*tfrecord*"
  --model "${MODEL_NAME}"
  --color_map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}"
  --class_weights="${CLASS_WEIGHTS_URL}"
  --channels="$CHANNEL_NAMES"
  --use_separate_channels $USE_SEPARATE_CHANNELS
  --batch_size $BATCH_SIZE
  --eval_set_size $EVAL_SET_SIZE
  --max_steps 0
  --base_loss $BASE_LOSS
  --seed $RANDOM_SEED
  --predict="$PREDICT_INPUT"
  --predict-output="$PREDICT_OUTPUT"
  ${TRAINING_ARGS[@]}
)

if [ "$USE_MODEL_EXPORT_PATH" == true ]; then
  COMMON_ARGS=(
    ${COMMON_ARGS[@]}
    --model_export_path="${MODEL_EXPORT_PATH}"
  )
fi

if [ $USE_SEPARATE_CHANNELS == true ]; then
  COMMON_ARGS=(
    ${COMMON_ARGS[@]}
    --color_map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}"
  )
fi

if [ $USE_CLOUD == true ]; then
  JOB_ID="predict_$JOB_ID"
  gcloud ml-engine jobs submit training "$JOB_ID" \
    --stream-logs \
    --module-name sciencebeam_gym.trainer.task \
    --package-path sciencebeam_gym \
    --staging-bucket "$TEMP_BUCKET" \
    --region us-central1 \
    --runtime-version=1.2 \
    -- \
    ${COMMON_ARGS[@]}
else
  gcloud ml-engine local train \
    --module-name sciencebeam_gym.trainer.task \
    --package-path sciencebeam_gym.trainer \
    -- \
    ${COMMON_ARGS[@]}
fi

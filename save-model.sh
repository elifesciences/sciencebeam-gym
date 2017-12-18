#!/bin/bash

source prepare-shell.sh

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
  --save_model="${MODEL_EXPORT_PATH}"
  ${TRAINING_ARGS[@]}
)

if [ ! -z "$QUALITATIVE_FOLDER_NAME" ]; then
  COMMON_ARGS=(
    ${COMMON_ARGS[@]}
    --qualitative_data_paths "${PREPROC_PATH}/${QUALITATIVE_FOLDER_NAME}/*tfrecord*"
    --qualitative_set_size ${QUALITATIVE_SET_SIZE}
  )
fi

if [ $USE_SEPARATE_CHANNELS == true ]; then
  COMMON_ARGS=(
    ${COMMON_ARGS[@]}
    --color_map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}"
  )
fi

if [ $USE_CLOUD == true ]; then
  JOB_ID="save_$JOB_ID"
  gcloud ml-engine jobs submit training "$JOB_ID" \
    --stream-logs \
    --module-name sciencebeam_gym.trainer.task \
    --package-path sciencebeam_gym \
    --staging-bucket "$BUCKET" \
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

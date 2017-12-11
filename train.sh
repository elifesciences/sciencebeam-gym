#!/bin/bash

export JOB_ID_FILE='.job-id'
if [ -f "$JOB_ID_FILE" ]; then
  rm "${JOB_ID_FILE}"
fi

source prepare-shell.sh

echo "TRAIN_PREPROC_TRAIN_PATH: $TRAIN_PREPROC_PATH"
echo "EVAL_PREPROC_EVAL_PATH: $EVAL_PREPROC_PATH"
echo "QUALITATIVE_PREPROC_EVAL_PATH: $QUALITATIVE_PREPROC_PATH"
echo "TRAIN_MODEL_PATH: $TRAIN_MODEL_PATH"
echo "CLASS_WEIGHTS_URL: ${CLASS_WEIGHTS_URL}"

COMMON_ARGS=(
  --output_path "${TRAIN_MODEL_PATH}/"
  --train_data_paths "${TRAIN_PREPROC_PATH}/*tfrecord*"
  --eval_data_paths "${EVAL_PREPROC_PATH}/*tfrecord*"
  --model "${MODEL_NAME}"
  --color_map "${CONFIG_PATH}/${COLOR_MAP_FILENAME}"
  --channels="$CHANNEL_NAMES"
  --class_weights="${CLASS_WEIGHTS_URL}"
  --use_separate_channels $USE_SEPARATE_CHANNELS
  --batch_size $BATCH_SIZE
  --eval_set_size $EVAL_SET_SIZE
  --seed $RANDOM_SEED
  --base_loss $BASE_LOSS
  ${TRAINING_ARGS[@]}
)

if [ ! -z "$QUALITATIVE_PREPROC_PATH" ]; then
  COMMON_ARGS=(
    ${COMMON_ARGS[@]}
    --qualitative_data_paths "${QUALITATIVE_PREPROC_PATH}/*tfrecord*"
    --qualitative_set_size ${QUALITATIVE_SET_SIZE}
  )
fi

if [ $USE_CLOUD == true ]; then
  echo "MAX_TRAIN_STEPS: $MAX_TRAIN_STEPS"
  gcloud ml-engine jobs submit training "$JOB_ID" \
    --stream-logs \
    --module-name sciencebeam_gym.trainer.task \
    --package-path sciencebeam_gym \
    --staging-bucket "$BUCKET" \
    --region us-central1 \
    --runtime-version=1.2 \
    --scale-tier=BASIC_GPU \
    -- \
    --save_max_to_keep 10 \
    --log_interval_secs 100000 \
    --eval_interval_secs 100000 \
    --save_interval_secs 100000 \
    --log_freq 500 \
    --eval_freq 500 \
    --save_freq 500 \
    --max_steps ${MAX_TRAIN_STEPS} \
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

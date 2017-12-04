source prepare-shell.sh

IMAGE_WIDTH=256
IMAGE_HEIGHT=256

if [ $USE_CLOUD == true ]; then
  CLOUD_ARGS="--cloud"
else
  CLOUD_ARGS=""
fi

python -m sciencebeam_gym.preprocess.preprocessing_pipeline \
  --data-path="$DATA_SOURCE_PATH" \
  --pdf-xml-file-list="$FILE_LIST_PATH/file-list-train.tsv" \
  --limit=$TRAIN_FILE_LIMIT \
  --output-path="${TRAIN_PREPROC_PATH}/" \
  --annotation-evaluation-csv="annotation-evaluation.tsv" \
  --xml-mapping-path=annot-xml-full.conf \
  --min-annotation-percentage=$MIN_ANNOTATION_PERCENTAGE \
  --save-svg \
  --save-tfrecords \
  --image-width ${IMAGE_WIDTH} \
  --image-height ${IMAGE_HEIGHT} \
  --pages=$PAGE_RANGE \
  --num_workers 1 \
  $CLOUD_ARGS

python -m sciencebeam_gym.preprocess.preprocessing_pipeline \
  --data-path="$DATA_SOURCE_PATH" \
  --pdf-xml-file-list="$FILE_LIST_PATH/file-list-validation.tsv" \
  --limit=$EVAL_FILE_LIMIT \
  --output-path="${EVAL_PREPROC_PATH}/" \
  --annotation-evaluation-csv="annotation-evaluation.tsv" \
  --xml-mapping-path=annot-xml-full.conf \
  --min-annotation-percentage=$MIN_ANNOTATION_PERCENTAGE \
  --save-svg \
  --save-tfrecords \
  --image-width ${IMAGE_WIDTH} \
  --image-height ${IMAGE_HEIGHT} \
  --pages=$PAGE_RANGE \
  --num_workers 1 \
  $CLOUD_ARGS

if [ ! -z "$QUALITATIVE_PREPROC_PATH" ]; then
  python -m sciencebeam_gym.preprocess.preprocessing_pipeline \
    --data-path="$DATA_SOURCE_PATH" \
    --pdf-xml-file-list="$FILE_LIST_PATH/file-list-validation.tsv" \
    --limit=$QUALITATIVE_FILE_LIMIT \
    --output-path="${QUALITATIVE_PREPROC_PATH}/" \
    --annotation-evaluation-csv="annotation-evaluation.tsv" \
    --xml-mapping-path=annot-xml-full.conf \
    --min-annotation-percentage=$MIN_ANNOTATION_PERCENTAGE \
    --save-svg \
    --save-tfrecords \
    --image-width ${IMAGE_WIDTH} \
    --image-height ${IMAGE_HEIGHT} \
    --pages=$PAGE_RANGE \
    --num_workers 1 \
    $CLOUD_ARGS
fi

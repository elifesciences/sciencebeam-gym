# ScienceBeam Gym Tools

## Image Annotation

### Find Bounding Boxes

For image training and evaluation it can be useful to know the bounding boxes of elements
in the rendered PDF pages, including for figure images.
As JATS XML is not designed to be tight to a particular presention,
one wouldn't usually find bounding boxes related to the rendered PDF in it.
Instead, it will link to graphic elements containing the figure image.

The JSON file aims to follow the [format of the COCO dataset](https://cocodataset.org/#format-data).

This command helps to determine the bounding boxes based on the linked graphic elements.

```bash
python -m sciencebeam_gym.tools.image_annotation.find_bounding_boxes \
  --pdf-file=/path/to/article.pdf \
  --image-files \
    /path/to/figure1.jpg \
    /path/to/figure2.jpg \
  --output-json-file=./bounding-box.json
```

Or using all of the graphic elements inside JATS XML:

```bash
python -m sciencebeam_gym.tools.image_annotation.find_bounding_boxes \
  --pdf-file=/path/to/article.pdf \
  --xml-file /path/to/article.xml \
  --output-json-file=./bounding-box.json
```

To visualize the bounding boxes, one may also add the `--output-annotated-images-path` option:

```bash
python -m sciencebeam_gym.tools.image_annotation.find_bounding_boxes \
  --pdf-file=/path/to/article.pdf \
  --xml-file /path/to/article.xml \
  --output-json-file=./bounding-box.json \
  --output-annotated-images-path=./annotated-pages/
```

Since the format follows the JSON format, other visualizing tools may work as well.

## Vocabulary

### Extract Vocabulary from Embeddings

```bash
python -m sciencebeam_gym.tools.vocabulary.extract_embeddings_vocabulary \
  --input-file=https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.gz \
  --output-vocabulary-file=./glove.6B.50d.vocab.txt
```

### Extract Vocabulary from XML Documents

```bash
python -m sciencebeam_gym.tools.vocabulary.create_vocabulary \
  --input-file-list=/path/to/file-list.lst \
  --output-word-count-file=/path/to/wordcount.tsv.gz \
  --sort-by-count \
  --limit=6000
```

### Filter Embeddings Vocabulary

```bash
python -m sciencebeam_gym.tools.vocabulary.filter_embeddings_vocabulary \
  --input-embeddings-file=https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.gz \
  --word-count-file=/path/to/wordcount.tsv.gz \
  --output-embeddings-file=/path/to/glove.6B.50d.filtered.txt
```

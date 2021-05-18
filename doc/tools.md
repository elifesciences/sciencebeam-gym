# ScienceBeam Gym Tools

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

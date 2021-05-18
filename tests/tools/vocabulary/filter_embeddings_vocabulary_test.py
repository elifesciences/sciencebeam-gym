from pathlib import Path
from sciencebeam_gym.tools.vocabulary.create_vocabulary import LOGGER

from sciencebeam_gym.tools.vocabulary.filter_embeddings_vocabulary import (
    iter_filter_embeddings_lines,
    main
)


class TestIterFilterEmbeddingsLines:
    def test_should_filter_tokens(self):
        assert list(iter_filter_embeddings_lines(
            ['token1 1 2 3\n', 'token2 2 3 4\n'],
            {'token1'}
        )) == [
            'token1 1 2 3\n'
        ]

    def test_should_include_tokens_case_insensitive(self):
        assert list(iter_filter_embeddings_lines(
            ['Token1 1 2 3\n', 'token2 2 3 4\n'],
            {'tOken1'}
        )) == [
            'Token1 1 2 3\n'
        ]


class TestMainEndToEnd:
    def test_should_filter_vocabulary_based_on_word_counts(
        self,
        tmp_path: Path
    ):
        input_embeddings_path = tmp_path / 'input-embedding.txt'
        word_count_path = tmp_path / 'word-count.tsv'
        output_embeddings_path = tmp_path / 'output-embedding.txt'
        input_embeddings_path.write_text('\n'.join([
            'token1 0 1 2',
            'token2 1 2 3',
            'token3 2 3 4'
        ]))
        word_count_path.write_text('\n'.join([
            'token\tcount',
            'token1\t1',
            'token3\t1'
        ]))
        main([
            '--input-embeddings-file=%s' % input_embeddings_path,
            '--word-count-file=%s' % word_count_path,
            '--output-embeddings-file=%s' % output_embeddings_path
        ])
        assert output_embeddings_path.exists()
        LOGGER.debug('output_embeddings_path: %r', output_embeddings_path.read_text())
        assert output_embeddings_path.read_text().splitlines() == [
            'token1 0 1 2',
            'token3 2 3 4'
        ]

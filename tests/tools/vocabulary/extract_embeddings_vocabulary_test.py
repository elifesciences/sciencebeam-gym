import gzip
from pathlib import Path

from sciencebeam_gym.tools.vocabulary.extract_embeddings_vocabulary import (
    main
)


class TestMainEndToEnd:
    def test_should_extract_vocabulary_from_embeddings_file(self, tmp_path: Path):
        embeddings_file = tmp_path / 'test.text'
        embeddings_file.write_text('\n'.join([
            'token1 0 1 2',
            'token2 1 2 3',
            'token3 2 3 4'
        ]))
        vocabulary_file = tmp_path / 'vocabulary.txt'
        main([
            f'--input-file={embeddings_file}',
            f'--output-vocabulary-file={vocabulary_file}'
        ])
        assert vocabulary_file.exists()
        assert vocabulary_file.read_text().splitlines() == [
            'token1', 'token2', 'token3'
        ]

    def test_should_extract_vocabulary_from_gz_embeddings_file(self, tmp_path: Path):
        embeddings_file = tmp_path / 'test.text.gz'
        with gzip.open(embeddings_file, 'w') as fp:
            fp.write('\n'.join([
                'token1 0 1 2',
                'token2 1 2 3',
                'token3 2 3 4'
            ]).encode('utf-8'))
        vocabulary_file = tmp_path / 'vocabulary.txt'
        main([
            f'--input-file={embeddings_file}',
            f'--output-vocabulary-file={vocabulary_file}'
        ])
        assert vocabulary_file.exists()
        assert vocabulary_file.read_text().splitlines() == [
            'token1', 'token2', 'token3'
        ]

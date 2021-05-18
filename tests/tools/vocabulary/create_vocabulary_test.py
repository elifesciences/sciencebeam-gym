from pathlib import Path

import pandas as pd

from sciencebeam_gym.tools.vocabulary.create_vocabulary import (
    iter_tokenized_tokens,
    main
)


class TestIterTokenizedTokens:
    def test_should_return_empty_iterable_for_whitespace_only_str(self):
        assert list(iter_tokenized_tokens(' \n\r ')) == []

    def test_should_ignore_surrounding_whitespace(self):
        assert list(iter_tokenized_tokens(' \n\rtoken1\n\r')) == [
            'token1'
        ]

    def test_should_split_on_space(self):
        assert list(iter_tokenized_tokens('token1 token2')) == [
            'token1', 'token2'
        ]

    def test_should_split_on_linefeed(self):
        assert list(iter_tokenized_tokens('token1\ntoken2')) == [
            'token1', 'token2'
        ]

    def test_should_split_on_comma_before_space_or_end(self):
        assert list(iter_tokenized_tokens('token1, token2,')) == [
            'token1', ',', 'token2', ','
        ]

    def test_should_split_on_dot_before_space_or_end(self):
        assert list(iter_tokenized_tokens('token1. token2.')) == [
            'token1', '.', 'token2', '.'
        ]


class TestMainEndToEnd:
    def test_should_extract_vocabulary_from_single_xml_file(self, tmp_path: Path):
        xml_file = tmp_path / 'test.xml'
        xml_file.write_text(
            '''
            <xml>
            <item>some text</item>
            <item>more text</item>
            </xml>
            '''
        )
        word_count_file = tmp_path / 'word-count.tsv'
        main([
            f'--input-file={xml_file}',
            f'--output-word-count-file={word_count_file}'
        ])
        assert word_count_file.exists()
        df = pd.read_csv(word_count_file, sep='\t')
        assert df.groupby('token').sum()['count'].to_dict() == {
            'text': 2,
            'some': 1,
            'more': 1
        }

    def test_should_extract_vocabulary_from_multiple_files_by_file_list_lst(
        self, tmp_path: Path
    ):
        xml_file_1 = tmp_path / 'test1.xml'
        xml_file_1.write_text('<xml><item>some text</item></xml>')
        xml_file_2 = tmp_path / 'test2.xml'
        xml_file_2.write_text('<xml><item>more text</item></xml>')
        file_list_file = tmp_path / 'file.lst'
        file_list_file.write_text('\n'.join([
            str(xml_file_1),
            str(xml_file_2)
        ]))
        word_count_file = tmp_path / 'word-count.tsv'
        main([
            f'--input-file-list={file_list_file}',
            f'--output-word-count-file={word_count_file}'
        ])
        assert word_count_file.exists()
        df = pd.read_csv(word_count_file, sep='\t')
        assert df.groupby('token').sum()['count'].to_dict() == {
            'text': 2,
            'some': 1,
            'more': 1
        }

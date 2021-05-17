from pathlib import Path

import pandas as pd

from sciencebeam_gym.preprocess.create_vocabulary_pipeline import (
    main
)


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

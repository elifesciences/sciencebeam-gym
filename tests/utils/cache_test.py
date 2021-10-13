from typing import Sequence

import pytest

from sciencebeam_gym.utils.cache import (
    MultiLevelCache,
    SimpleDictCache
)


KEY_1 = 'key1'
VALUE_1 = 'value1'


@pytest.fixture(name='dict_cache_list')
def _dict_cache_list() -> Sequence[SimpleDictCache]:
    return [SimpleDictCache() for _ in range(10)]


class TestMultiLevelCache:
    def test_should_return_none_if_not_in_any_cache(
        self,
        dict_cache_list: Sequence[SimpleDictCache]
    ):
        cache = MultiLevelCache(dict_cache_list[:2])
        assert cache.get(KEY_1) is None

    def test_should_populate_cache_and_retrieve_item(
        self,
        dict_cache_list: Sequence[SimpleDictCache]
    ):
        cache = MultiLevelCache(dict_cache_list[:2])
        cache[KEY_1] = VALUE_1
        assert cache.get(KEY_1) == VALUE_1

    def test_should_retrieve_item_from_next_level(
        self,
        dict_cache_list: Sequence[SimpleDictCache]
    ):
        cache = MultiLevelCache(dict_cache_list[:3])
        cache[KEY_1] = VALUE_1
        del dict_cache_list[0][KEY_1]
        del dict_cache_list[2][KEY_1]
        assert cache.get(KEY_1) == VALUE_1

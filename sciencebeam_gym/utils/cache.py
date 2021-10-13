from typing import Optional, Sequence, TypeVar

from typing_extensions import Protocol


T_Key = TypeVar('T_Key', contravariant=True)
T_Value = TypeVar('T_Value')


class CacheProtocol(Protocol[T_Key, T_Value]):
    def get(self, key: T_Key) -> Optional[T_Value]:
        pass

    def __setitem__(self, key: T_Key, value: T_Value):
        pass

    def __delitem__(self, key: T_Key):
        pass


class SimpleDictCache(CacheProtocol):
    def __init__(self):
        super().__init__()
        self._data = {}

    def get(self, key: T_Key) -> Optional[T_Value]:
        return self._data.get(key)

    def __setitem__(self, key: T_Key, value: T_Value):
        self._data[key] = value

    def __delitem__(self, key: T_Key):
        del self._data[key]


class MultiLevelCache(CacheProtocol):
    def __init__(self, cache_list: Sequence[CacheProtocol]):
        super().__init__()
        self.cache_list = cache_list

    def get(self, key: T_Key) -> Optional[T_Value]:
        for cache in self.cache_list:
            value = cache.get(key)
            if value is not None:
                return value
        return None

    def __setitem__(self, key: T_Key, value: T_Value):
        for cache in self.cache_list:
            cache[key] = value

    def __delitem__(self, key: T_Key):
        for cache in self.cache_list:
            try:
                del cache[key]
            except KeyError:
                pass

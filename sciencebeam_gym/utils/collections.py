from typing import Dict, TypeVar


K = TypeVar('K')
V = TypeVar('V')


def get_inverted_dict(d: Dict[K, V]) -> Dict[V, K]:
    return {v: k for k, v in d.items()}

from functools import lru_cache

from .transit.transitous import TransitousProvider


@lru_cache
def get_transit() -> TransitousProvider:
    return TransitousProvider()

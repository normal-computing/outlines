import os
from typing import Callable

from perscache import Cache, NoCache
from perscache.serializers import JSONSerializer
from perscache.storage import LocalFileStorage

home_dir = os.path.expanduser("~")
cache_dir = os.environ.get("OUTLINES_CACHE_DIR", f"{home_dir}/.cache/outlines")
memory = Cache(serializer=JSONSerializer(), storage=LocalFileStorage(cache_dir))


def cache(fn: Callable):
    return memory.cache()(fn)


def get_cache():
    """Get the context object that contains previously-computed return values.

    The cache is used to avoid unnecessary computations and API calls, which can
    be long and expensive for large models.

    The cache directory defaults to `HOMEDIR/.cache/outlines`, but this choice
    can be overriden by the user by setting the value of the `OUTLINES_CACHE_DIR`
    environment variable.

    """
    return memory


def disable_cache():
    """Disable the cache for this session.

    Generative models output different results each time they are called when
    sampling. This can be a desirable property for some workflows, in which case
    one can call `outlines.call.disable` to disable the cache for the session.

    This function does not delete the cache, call `outlines.cache.clear`
    instead. It also does not overwrite the cache with the values returned
    during the session.

    Example
    -------

    `outlines.cache.disable` should be called right after importing outlines:

    >>> import outlines.cache as cache
    >>> cache.disable()

    """
    global memory
    memory = NoCache()


def clear_cache():
    """Erase the cache completely."""
    global memory
    memory.storage.clear()


def enable_cache():
    """Enable the cache."""
    global memory

    if isinstance(memory, NoCache):
        memory = Cache(serializer=JSONSerializer(), storage=LocalFileStorage(cache_dir))

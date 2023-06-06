"""Outlines is a Generative Model Programming Framework."""
import random
from contextlib import suppress

import numpy as np

from outlines.caching import clear_cache, disable_cache, enable_cache, get_cache
from outlines.text import prompt

__all__ = [
    "clear_cache",
    "disable_cache",
    "get_cache",
    "enable_cache",
    "prompt",
]


def set_seed(seed: int):
    """Set a global seed for output generation.

    This highly undesirable approach is taken from
    https://github.com/huggingface/transformers/blob/118e9810687dd713b6be07af79e80eeb1d916908/src/transformers/trainer_utils.py#L83
    """
    random.seed(seed)
    np.random.seed(seed)

    with suppress(ImportError):
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

    with suppress(ImportError):
        import tensorflow

        tensorflow.random.set_seed(seed)

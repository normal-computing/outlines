import outlines

outlines.disable_cache()

import numpy as np  # noqa
import pytest  # noqa

from outlines.models.hf_transformers import (  # noqa
    HuggingFaceCompletion,
    HuggingFaceEmbeddings,
)

TEST_MODEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


def test_samples():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)

    answer = model("test", samples=1)
    assert isinstance(answer, str)

    answer = model("test")
    assert isinstance(answer, str)

    answers = model("test", samples=3)
    assert isinstance(answers, list)
    assert len(answers) == 3


def test_type_int():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    answer = model("test", type="int")
    int(answer)


def test_type_float():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    answer = model("test", type="float")
    float(answer)


def test_incompatible_constraints():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)

    with pytest.raises(ValueError):
        model("test", type="float", is_in=["test"])


def test_choices():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=50)

    choices = ["a", "and a long sequence", "with\n line break"]
    answer = model("test", is_in=choices)
    assert answer in choices


def test_stop():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=1000)

    stop = [" ", "\n"]
    answer = model("test", stop_at=stop)
    for seq in stop:
        assert seq not in answer


@pytest.mark.xfail
def test_type_multiple_samples():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    answer = model("test", type="int", samples=2)
    int(answer)


@pytest.mark.xfail
def test_is_in_multiple_samples():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    model("test", is_in=["a", "b"], samples=2)


@pytest.mark.xfail
def test_stop_at_multiple_samples():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)
    model("test", stop_at=[" "], samples=2)


def test_single_embedding():
    model = HuggingFaceEmbeddings(EMBED_MODEL)
    out = model("This is a test sentence")
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 1
    assert out.shape[1] == 384


def test_multiple_embedding():
    model = HuggingFaceEmbeddings(EMBED_MODEL)
    out = model(["This is a test sentence", "Another test sentence"])
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 2
    assert out.shape[1] == 384


def test_batch_embeddings():
    model = HuggingFaceEmbeddings(EMBED_MODEL)
    out = model(["This is a test sentence", "Another test sentence"], batch_size=4)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 2
    assert out.shape[1] == 384

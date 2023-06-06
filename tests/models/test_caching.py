import outlines
from outlines.models.hf_transformers import HuggingFaceCompletion  # noqa

TEST_MODEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"


def test_seed_with_caching():
    model = HuggingFaceCompletion(TEST_MODEL, max_tokens=10)

    outlines.set_seed(48209)

    answer1 = model("test")

    outlines.set_seed(48209)

    answer2 = model("test")
    assert answer2 == answer1

    answer3 = model("test")
    assert answer3 == answer2

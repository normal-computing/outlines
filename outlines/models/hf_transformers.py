"""Integration with HuggingFace's `transformers` library."""
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from outlines.caching import cache

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


def HuggingFaceCompletion(
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Callable:
    """Create a function that will call the `generate` method of a `transformers` model.

    You should have the `torch` and `transformers` packages installed. First
    execution may take a while since the pre-trained weights will be downloaded.
    Available models are listed on `HuggingFace's model page <https://huggingface.co/models>`_.

    Note
    ----

    To my knowledge `tranformers` does not simply allow to stop the generation
    after a given sequence has been generated. We will need to implement this
    manually for this integration to have the same features as `OpenAICompletion`.

    Parameters
    ----------
    model_name: str
        The name of the model as listed on HuggingFace's models page.
    max_tokens
        The maximum number of tokens to generate.
    temperature
        Value used to module the next token probabilities.

    Returns
    -------
    A function that will generate tokens from the model when passed a prompt.

    """
    if max_tokens is None:
        max_tokens = 216

    if temperature is None:
        temperature = 1.0

    def call(
        prompt: str,
        *,
        samples: int = 1,
        stop_at: Optional[List[str]] = None,
        is_in: Optional[List[str]] = None,
        type: Optional[str] = None,
    ) -> str:
        return call_model_generate_method(
            model_name, prompt, max_tokens, temperature, samples, stop_at, is_in, type
        )

    return call


@cache
def call_model_generate_method(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    samples: int,
    stop_at: List[str],
    is_in: List[str],
    type: str,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt_tokens = tokenizer(prompt, return_tensors="pt")

    logit_processors: Optional[List[Callable]] = None
    stopping_criteria: Optional[List[Callable]] = None
    postprocessing: Callable = lambda x: x
    if type is not None:
        if samples > 1:
            raise NotImplementedError(
                "It is currently not possible to control the generation of several samples with the `transformers` integration"
            )
        if is_in is not None:
            raise ValueError(
                "You cannot both restrict to a set of choices with `is_in` and to a type with `type`"
            )
        logit_processor, stopping_criterion, postprocessing = create_type_constraint(
            type, tokenizer, prompt_tokens["input_ids"]
        )
        logit_processors = [logit_processor]
        stopping_criteria = [stopping_criterion]
    elif is_in is not None:
        if samples > 1:
            raise NotImplementedError(
                "It is currently not possible to control the generation of several samples with the `transformers` integration"
            )
        if stop_at is not None:
            raise ValueError(
                "You cannot both restrict to a set of choices with `is_in` and set a stopping criterion"
            )
        logit_processor, stopping_criterion, postprocessing = create_choice_constraint(
            is_in, tokenizer, prompt_tokens["input_ids"]
        )
        logit_processors = [logit_processor]
        stopping_criteria = [stopping_criterion]
    elif stop_at is not None:
        if samples > 1:
            raise NotImplementedError(
                "It is currently not possible to control the generation of several samples with the `transformers` integration"
            )
        logit_processor, stopping_criterion, postprocessing = create_stop_constraint(
            stop_at, tokenizer, prompt_tokens["input_ids"]
        )
        logit_processors = [logit_processor]
        stopping_criteria = [stopping_criterion]

    if torch.cuda.is_available():
        model = model.to("cuda")
        prompt_tokens = prompt_tokens.to("cuda")

    returned_tokens = model.generate(
        **prompt_tokens,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=samples,
        logits_processor=logit_processors,
        stopping_criteria=stopping_criteria,
    )
    new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] :]
    new_tokens = new_tokens.squeeze()

    if samples == 1:
        results = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results = postprocessing(results)
    else:
        results = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return results


def create_stop_constraint(
    stop_at: List[str],
    tokenizer: "PreTrainedTokenizerBase",
    prompt_tokens: "torch.Tensor",
) -> Tuple[Callable, Callable, Callable]:
    """Create a constraint that stops generation after a sequence has been found.

    Parameters
    ----------
    stop_at
        The list of sequences which, once generated, the generation is stopped.
    tokenizer
        The tokenizer that corresponds to the model used for generation.
    prompt_tokens
        An array that contains the tokenized prompt.

    """
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]

    def stopping_criterion(input_ids: torch.Tensor, _) -> bool:
        """Choose whether to stop the generation after this step.

        We check whether either of the stopping sequences is present in the
        current generation. If either one is found we stop the generation.

        """
        decoded_input = tokenizer.decode(
            input_ids[0, num_prompt_tokens:], skip_special_tokens=True
        )
        for stopping_sequence in stop_at:
            if stopping_sequence in decoded_input:
                return True

        return False

    def postprocess(generated_sequence: str) -> str:
        """Postprocess the generated text.

        We need to remove the stopping sequence that triggered the end of
        the generation at the end.

        """
        for stopping_sequence in stop_at:
            idx = generated_sequence.find(stopping_sequence)
            if idx != -1:
                return generated_sequence[:idx]

        return generated_sequence

    return lambda _, x: x, stopping_criterion, postprocess


def create_choice_constraint(
    choices: List[str],
    tokenizer: "PreTrainedTokenizerBase",
    prompt_tokens: "torch.Tensor",
) -> Tuple[Callable, Callable, Callable]:
    """Create a constraint that forces the generation to be among a list of choices.

    Parameters
    ----------
    choices
        The list of sequences to which the generated sequences must belong.
    tokenizer
        The tokenizer that corresponds to the model used for generation.
    prompt_tokens
        An array that contains the tokenized prompt.

    """
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]
    tokenized_choices = [tokenizer.encode(word) for word in choices]

    def logit_processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Pre-process the model's output logits before generating the next token.

        At each step we forbid the tokens that do not steer the generation in the
        direction of being either of the choices.

        """
        output = input_ids[0, num_prompt_tokens:]
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)

        mask = torch.zeros(len(tokenizer), dtype=torch.bool)
        for choice, tokens in zip(choices, tokenized_choices):
            if not choice.startswith(decoded_output):
                continue
            else:
                mask[tokens[len(output)]] = True

        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")

        return scores

    def stopping_criterion(input_ids: torch.Tensor, _) -> bool:
        """Choose whether to stop the generation after this step.

        We stop generation when either of the choices has been found.

        TODO: We can stop the generation once we have excluded all possibilities
        but one, and the full sequence can be recovered during post-processing.

        """
        decoded_input = tokenizer.decode(
            input_ids[0, num_prompt_tokens:], skip_special_tokens=True
        )

        is_present_in_output = []
        for choice in choices:
            if choice == decoded_input:
                return True
            elif choice.startswith(decoded_input):
                is_present_in_output.append(1)
            else:
                is_present_in_output.append(0)

        # If we have eliminated all possibilities but one, return
        if sum(is_present_in_output) == 1:
            return True

        return False

    def postprocess(output_sequence: str) -> str:
        for choice in choices:
            if choice.startswith(output_sequence):
                return choice

        return output_sequence

    return logit_processor, stopping_criterion, postprocess


def create_int_constraint(
    tokenizer: "PreTrainedTokenizerBase", prompt_tokens: "torch.Tensor"
) -> Tuple[Callable, Callable, Callable]:
    """Create a constraints that forces the generated sequence to be an integer.

    Parameters
    ----------
    tokenizer
        The tokenizer that corresponds to the model used for generation.
    prompt_tokens
        An array that contains the tokenized prompt.

    """
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]

    mask = torch.zeros(len(tokenizer), dtype=torch.bool)

    for _, token_id in tokenizer.get_vocab().items():
        token = tokenizer.decode(token_id)
        are_all_digits = all([c.isdigit() for c in token])
        if are_all_digits:
            mask[token_id] = True

    mask[tokenizer.eos_token_id] = False

    def logit_processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Pre-process the model's output logits before generating the next token.

        At each step we forbid the tokens that do not correspond to a digit. We forbid
        EOS tokens until at least one digit has been generated.

        # TODO: Do we need to allow " ", "\n", "\r" and other delimiters?

        """
        if input_ids.shape[1] > num_prompt_tokens + 1:
            mask[tokenizer.eos_token_id] = True
        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")
        return scores

    return logit_processor, lambda *_: False, lambda x: x


def create_float_constraint(
    tokenizer: "PreTrainedTokenizerBase",
    prompt_tokens: "torch.Tensor",
    decimals: int = 3,
) -> Tuple[Callable, Callable, Callable]:
    """Create a constraints that forces the generated sequence to be an floating point number.

    Parameters
    ----------
    tokenizer
        The tokenizer that corresponds to the model used for generation.
    prompt_tokens
        An array that contains the tokenized prompt.

    """
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]

    mask = torch.zeros(len(tokenizer), dtype=torch.bool)

    for _, token_id in tokenizer.get_vocab().items():
        token = tokenizer.decode(token_id)
        is_valid_float_or_int = (
            all([c.isdigit() or c == "." for c in token]) and token.count(".") <= 1
        )
        if is_valid_float_or_int:
            mask[token_id] = True

    mask[tokenizer.eos_token_id] = False

    def logit_processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Pre-process the model's output logits before generating the next token.

        At each step we forbid the tokens that do not correspond to a digit. We forbid
        EOS tokens until at least one digit has been generated.

        # TODO: Do we need to allow " ", "\n", "\r" and other delimiters?

        """
        if input_ids.shape[1] > num_prompt_tokens + 1:
            mask[tokenizer.eos_token_id] = True
        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")
        return scores

    def stopping_criterion(input_ids: torch.Tensor, _) -> bool:
        """Choose whether to stop the generation after this step.

        We stop generation if the sequence contains more than one period, or
        if the desired number of decimals has been generated.

        """
        decoded_input = tokenizer.decode(
            input_ids[0, num_prompt_tokens:], skip_special_tokens=True
        )
        if decoded_input.count(".") > 1:
            return True

        if (
            decoded_input.count(".") == 1
            and len(decoded_input.strip().split(".")[1]) > decimals
        ):
            return True

        return False

    def postprocessing(output: str) -> str:
        """Postprocess the generated text.

        We need to remove the trailing period, present if the generation
        was stopped because a second period was found.

        """
        return output.rstrip(".")

    return logit_processor, stopping_criterion, postprocessing


type_to_mask: Dict[str, Callable] = {
    "float": create_float_constraint,
    "int": create_int_constraint,
}


def create_type_constraint(
    type: str, tokenizer: "PreTrainedTokenizerBase", prompt_tokens: "torch.Tensor"
) -> Tuple[Callable, Callable, Callable]:
    if type not in ["int", "float"]:
        raise NotImplementedError(f"Cannot restrict the generation to type {type}")

    return type_to_mask[type](tokenizer, prompt_tokens)


def HuggingFaceEmbeddings(model_name: str):
    """Create a function that will download and run a HuggingFace embedding locally.

    You should have the `transformers` package installed. Available models are listed
    on the `HuggingFace documentation <https://huggingface.co/sentence-transformers`_.

    Note: The first time this is run it might take 20-30 seconds to download the model weights.

    Parameters
    ----------
    model_name: str
        The model name as listed in the HuggingFace website.

    Returns
    -------
    A function that will call run the HuggingFace embedding with the given parameters when
    passed a prompt. It will use a GPU if there is one available.

    """

    def mean_pooling(model_output, attention_mask):
        """Pools together the outputs from a sentence embedding model to generate a vector embedding."""
        import torch

        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embedding(
        sentences: Union[List[str], str], *, batch_size: Optional[int] = None
    ):
        """Generates embeddings from mean-pooled sentence embeddings from HuggingFace sentence transformers.

        Parameters
        ----------
        sentences
            The strings to be embedded
        batch_size: Optional[int]
        The batch size. If it is not provided, or if a negative value is given, the embeddings will be run as a single batch.


        Returns
        -------
        A function that will call run the HuggingFace embedding with the given parameters when
        passed a prompt. It will use a GPU if there is one available.

        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        # Set up sentence transformer (using appropriate resources)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.to("cuda")

        # Do some padding and batching logic
        if isinstance(sentences, str):
            sentences = [sentences]

        n_sentences = len(sentences)
        if batch_size is None:
            batch_size = n_sentences

        if n_sentences % batch_size != 0:
            pad = ((n_sentences // batch_size) + 1) * batch_size - n_sentences
            sentences.extend(["PAD"] * pad)

        embeddings_list = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]
            encoded_input = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            if torch.cuda.is_available():
                encoded_input = encoded_input.to("cuda")

            with torch.no_grad():
                model_output = model(**encoded_input)

            batch_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings, p=2, dim=1
            )

            embeddings_list.append(batch_embeddings)

        embeddings_tensor = torch.cat(embeddings_list, dim=0)
        return embeddings_tensor.cpu().numpy()[:n_sentences,]  # check slicing here

    return get_embedding

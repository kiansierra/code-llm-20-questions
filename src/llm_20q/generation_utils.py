import torch
from loguru import logger
from transformers import AutoModelForCausalLM


def update_input_ids(input_ids: torch.Tensor, next_token: int) -> torch.Tensor:
    selected_tensor = next_token * torch.ones((1, 1)).long().to(input_ids.device)
    return torch.cat([input_ids, selected_tensor], dim=-1)


def generate_options(model: AutoModelForCausalLM, sequences_ids: list[list[int]], input_ids: torch.Tensor) -> list[int]:
    """
    Chooses one of the  a sequence of options based on a given model, input sequences, and initial input_ids.

    Args:
        model (AutoModelForCausalLM): The model used for generating options.
        sequences_ids (list[list[int]]): A list of sequences represented as lists of token_ids.
        input_ids (torch.Tensor): The initial input_ids for generating options.

    Returns:
        list[int]: The generated sequence of options.

    Raises:
        RuntimeError: If no options are left.

    """

    step = 0
    past_key_values = None
    all_sequence_options = sequences_ids
    logger.info(f"step: {step},  {len(all_sequence_options)=}")
    while len(all_sequence_options) > 0:
        options = [elem[step] for elem in all_sequence_options]
        if len(set(options)) == 1:
            selected = options[0]
            input_ids = update_input_ids(input_ids, selected)
            step += 1
            continue
        output = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        next_token_logits = output.logits[0][-1][options].softmax(-1)
        next_token = options[next_token_logits.argmax()]
        input_ids = update_input_ids(input_ids, next_token)
        all_sequence_options = [elem for elem in all_sequence_options if elem[step] == next_token]
        step += 1
        logger.info(f"step: {step},  {len(all_sequence_options)=}")
        if len(all_sequence_options) == 1:
            return all_sequence_options[0]
        if len(all_sequence_options) == 0:
            raise RuntimeError("No options left")
    return all_sequence_options[0]

from collections import OrderedDict

import numpy as np
import torch

from .conversion import sequence2model

PREDICT_FUNC_DICT = OrderedDict()


def register_predict_func(func):
    key = func.__name__
    if key in PREDICT_FUNC_DICT:
        raise KeyError(f'A predict function key `{key}` has been already registered')

    PREDICT_FUNC_DICT[key] = func
    return func


@register_predict_func
def default_predict(logits, decoded_tokens, complete_flag_list, unavailable_var_indices, vocabulary, training):
    tmp_logits = logits.clone()
    # Mask logits with unavailable variables
    masked_indices = [0] + unavailable_var_indices
    tmp_logits[:, :, masked_indices] = -np.inf
    top_indices = torch.argmax(tmp_logits[:, -1:], dim=-1)
    for i in range(top_indices.shape[0]):
        if not training and complete_flag_list[i]:
            top_indices[i] = vocabulary.ignored_index

    decoded_tokens = torch.cat([decoded_tokens, top_indices], dim=-1)
    complete_flag_list = list()
    for i, token_indices in enumerate(decoded_tokens):
        if len(token_indices) == 1:
            complete_flag_list.append(False)
            continue

        # Skip the first element (<SOS>)
        symbols = vocabulary.convert_indices_to_symbols(token_indices.detach().tolist()[1:])
        try:
            _, parent_stack = sequence2model(symbols, returns_parent_stack=True)
            complete_flag_list.append(len(parent_stack) == 0)
        except IndexError:
            complete_flag_list.append(False)
    return decoded_tokens, logits, complete_flag_list


def get_predict_function(func_name, **kwargs):
    if func_name in PREDICT_FUNC_DICT:
        return PREDICT_FUNC_DICT[func_name]
    raise KeyError(f'func_name `{func_name}` is not expected')


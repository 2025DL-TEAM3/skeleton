import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Optional, Tuple, List, Dict, Any, Union, Set
import warnings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)

def get_or_map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:  # find and/or update special token mappings
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
        for v in data.values():  # recursively process dict values
            tokens.update(get_or_map_special_tokens(v, mapping))
    if isinstance(data, list):
        for v in data:  # recursively process lists
            tokens.update(get_or_map_special_tokens(v, mapping))
    return tokens


def remove_tokenizer_normalizer(tokenizer):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def shrink_tokenizer_vocab(
    tokenizer: PreTrainedTokenizer, 
    keep_token_ids: Set[int], 
    keep_special=True, 
    remove_unk=False
):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json['model']['type'] == "BPE"

    if keep_special:  # get special tokens to keep
        keep_token_ids.update(tokenizer.all_special_ids)
        keep_token_ids.update(get_or_map_special_tokens(tok_json.get('post_processor')))

    if remove_unk:  # remove unknown token
        keep_token_ids -= {tokenizer.unk_token_id}

    # build mapping from old to new id
    mapping = {old: new for new, old in enumerate(sorted(keep_token_ids))}

    # update tokenizer info
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = []
    tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
    tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])
    get_or_map_special_tokens(tok_json.get('post_processor'), mapping)

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))  # reload json, modifying tokenizer in-place

    if remove_unk:
        tokenizer.unk_token = None

    return mapping  # token mapping to be used later


def shrink_model_embeddings(
    model: PreTrainedModel,
    mapping: Dict[int, int] # key: old token id, value: new token id
):
    with torch.no_grad():
        # copy embeddings to keep
        old_token_row_indices = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        old_token_row_indices = old_token_row_indices.to(model.get_input_embeddings().weight.data.device)
        selected_rows_input = torch.index_select(model.get_input_embeddings().weight.data, 0, old_token_row_indices)
        old_token_row_indices = old_token_row_indices.to(model.get_output_embeddings().weight.data.device)
        selected_row_output = torch.index_select(model.get_output_embeddings().weight.data, 0, old_token_row_indices)

        # resize model embeddings
        model.resize_token_embeddings(len(old_token_row_indices))

        # set to copied values
        model.get_input_embeddings().weight.data[:] = selected_rows_input
        model.get_output_embeddings().weight.data[:] = selected_row_output

        # map model tokens to new id
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))

def apply_custom_head(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer,
):
    """
    1. prepare allowed token strings
    2. convert to token ids
    3. shrink tokenizer vocab
    """
    
    keep_token_ids = set()
    
    # digits
    for i in range(10):
        token_id = tokenizer.encode(str(i), add_special_tokens=False)[0]
        keep_token_ids.add(token_id)
    
    # Add thinking tokens
    try:
        think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        keep_token_ids.add(think_start_id)
        keep_token_ids.add(think_end_id)
    except Exception as e:
        warnings.warn(f"Could not add thinking tokens: {e}")
    
    # Add newline token for ARC grid formatting
    try:
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        keep_token_ids.add(newline_id)
    except Exception as e:
        warnings.warn(f"Could not add newline token: {e}")

    # Add space token for ARC grid formatting
    try:
        space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        keep_token_ids.add(space_id)
    except Exception as e:
        warnings.warn(f"Could not add space token: {e}")
    
    # Find other useful tokens for ARC
    try:
        tokens_to_check = ["[", "]", "{", "}", "(", ")", ":", ",", "."]
        for token in tokens_to_check:
            try:
                ids = tokenizer.encode(token, add_special_tokens=False)
                for token_id in ids:
                    keep_token_ids.add(token_id)
            except:
                pass
    except Exception as e:
        warnings.warn(f"Could not add utility tokens: {e}")
        
    # Add all tokens from the prompt templates
    try:
        # Import here to avoid circular import
        from .arc_utils import format_prompt_messages
        
        # Create a sample prompt to include all template parts
        sample_datapoint = {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[5, 6], [7, 8]],
                },
            ],
            "test": [
                {
                    "input": [[9, 8], [7, 6]],
                    "output": None,
                },
            ],
        }
        
        fomatted_point = format_prompt_messages(datapoint=sample_datapoint)
        
        prompt_chat_template_applied = tokenizer.apply_chat_template(
            fomatted_point, 
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,        
        )
        
        # Tokenize the sample prompt and add all token IDs to keep_tokens
        prompt_token_ids = tokenizer.encode(prompt_chat_template_applied, add_special_tokens=False)
        for token_id in prompt_token_ids:
            keep_token_ids.add(token_id)
            
        print(f"Added {len(prompt_token_ids)} tokens from prompt templates")
    except Exception as e:
        warnings.warn(f"Error adding prompt template tokens: {e}")
    
    # keep tokens used by model
    for config in [model.config, model.generation_config]:
        for k, v in config.to_dict().items():
            if k.endswith('token_id'):
                keep_token_ids.update(v if isinstance(v, list) else [v])
    keep_token_ids -= {None}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_token_ids, keep_special=True, remove_unk=True)
    shrink_model_embeddings(model, mapping)
    
    print(f"✓ Model vocabulary optimized for ARC: {len(mapping)} tokens kept")
    return model, tokenizer, mapping
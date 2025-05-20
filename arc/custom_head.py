import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Optional, Tuple, List, Dict, Any, Union, Set
import warnings

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


def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special=True, remove_unk=False):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json['model']['type'] == "BPE"

    if keep_special:  # get special tokens to keep
        keep_indices.update(tokenizer.all_special_ids)
        keep_indices.update(get_or_map_special_tokens(tok_json.get('post_processor')))

    if remove_unk:  # remove unknown token
        keep_indices -= {tokenizer.unk_token_id}

    # build mapping from old to new id
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

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


def shrink_model_embeddings(model, mapping):
    with torch.no_grad():
        # copy embeddings to keep
        row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select)
        row_select = row_select.to(model.get_output_embeddings().weight.data.device)
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select)

        # resize model embeddings
        model.resize_token_embeddings(len(row_select))

        # set to copied values
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head

        # map model tokens to new id
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))


def keep_single_char_tokens(model, tokenizer, keep=None, keep_norm=False, keep_model_tok=True, **kwargs):
    if not keep_norm:
        remove_tokenizer_normalizer(tokenizer)  # required for some models
    if keep is None:  # keep all single_length tokens
        keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    else:  # keep tokens that were passed
        keep_indices = set(tokenizer.vocab[t] for t in keep)
    if keep_model_tok:  # keep tokens used by model
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update(v if isinstance(v, list) else [v])
    keep_indices -= {None}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
    shrink_model_embeddings(model, mapping)
    return mapping

def apply_custom_head(model, tokenizer, keep_digits=True, keep_thinking_tokens=True, keep_special=True):
    """
    Optimize the model specifically for ARC tasks by reducing the vocabulary size
    to only what's needed for ARC (digits, thinking tokens, special tokens, and prompt templates).
    
    Args:
        model: The model to optimize
        tokenizer: The tokenizer to optimize
        keep_digits: Whether to keep digit tokens (0-9)
        keep_thinking_tokens: Whether to keep thinking tokens (<think>, </think>)
        keep_special: Whether to keep special tokens (eos, bos, pad, etc.)
        
    Returns:
        tuple: (model, tokenizer, id_mapping)
    """
    # Tokens to keep
    keep_tokens = set()
    
    # Add digit tokens (0-9)
    if keep_digits:
        for i in range(10):
            token_id = tokenizer.encode(str(i), add_special_tokens=False)[0]
            keep_tokens.add(token_id)
    
    # Add thinking tokens
    if keep_thinking_tokens:
        try:
            think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
            think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
            keep_tokens.add(think_start_id)
            keep_tokens.add(think_end_id)
        except Exception as e:
            warnings.warn(f"Could not add thinking tokens: {e}")
    
    # Add newline token for ARC grid formatting
    try:
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        keep_tokens.add(newline_id)
    except Exception as e:
        warnings.warn(f"Could not add newline token: {e}")

    # Add space token for ARC grid formatting
    try:
        space_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        keep_tokens.add(space_id)
    except Exception as e:
        warnings.warn(f"Could not add space token: {e}")
    
    # Find other useful tokens for ARC
    try:
        tokens_to_check = ["[", "]", "{", "}", "(", ")", ":", ",", ".", "input", "output"]
        for token in tokens_to_check:
            try:
                ids = tokenizer.encode(token, add_special_tokens=False)
                for token_id in ids:
                    keep_tokens.add(token_id)
            except:
                pass
    except Exception as e:
        warnings.warn(f"Could not add utility tokens: {e}")
        
    # Add all tokens from the prompt templates
    try:
        # Import here to avoid circular import
        from .arc_utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
        
        # Create a sample prompt to include all template parts
        sample_prompt = system_prompt
        sample_prompt += user_message_template1.format(
            n=2, 
            plural="s", 
            examples="Example 1 Input:\n0000\nExample 1 Output:\n1111\n----------------------------------------\nExample 2 Input:\n0123\nExample 2 Output:\n1234\n----------------------------------------\n"
        )
        sample_prompt += user_message_template2.format(test_grid="Test Input:\n0000")
        sample_prompt += user_message_template3
        
        # Tokenize the sample prompt and add all token IDs to keep_tokens
        prompt_token_ids = tokenizer.encode(sample_prompt, add_special_tokens=True)
        for token_id in prompt_token_ids:
            keep_tokens.add(token_id)
            
        print(f"Added {len(prompt_token_ids)} tokens from prompt templates")
    except Exception as e:
        warnings.warn(f"Error adding prompt template tokens: {e}")
    
    # Use the keep_single_char_tokens function for the actual optimization
    mapping = keep_single_char_tokens(
        model, 
        tokenizer, 
        keep=None,  # Let the function handle it based on our keep_tokens
        keep_norm=False, 
        keep_model_tok=True,
        keep_special=keep_special
    )
    
    print(f"✓ Model vocabulary optimized for ARC: {len(mapping)} tokens kept")
    return model, tokenizer, mapping
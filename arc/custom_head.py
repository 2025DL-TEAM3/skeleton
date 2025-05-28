import os
import json
from collections import OrderedDict
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

def indices_required_for_merges(keep_indices, vocab, merges):
    merges_lookup = {}
    for m in merges:
        a, b = m.split(' ') if isinstance(m, str) else m
        key = vocab[f'{a}{b}']
        if key not in merges_lookup: merges_lookup[key] = set()
        merges_lookup[key].add(vocab[a])
        merges_lookup[key].add(vocab[b])
    to_process = list(keep_indices)
    while len(to_process):
        for w in merges_lookup.get(to_process.pop(), []):
            if w not in keep_indices:
                keep_indices[w] = None
                to_process.append(w)
    return keep_indices

def remove_unused_merges(merges, vocab):
    return [f'{a} {b}' for a, b in [m.split(' ') if isinstance(m, str) else m for m in merges] if all(w in vocab for w in [a, b, a + b])]

def map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
    for v in (data.values() if isinstance(data, dict) else data if isinstance(data, list) else []):
        tokens.update(map_special_tokens(v, mapping))
    return tokens


def remove_tokenizer_normalizer(tokenizer):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def shrink_tokenizer_vocab(
    tokenizer: PreTrainedTokenizer, 
    keep_indices: OrderedDict[int, None], 
    keep_special=True, 
    keep_token_order=False
):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json['model']['type'] == "BPE"

    if keep_special:  # get special tokens to keep
        keep_indices.update({k: None for k in tokenizer.all_special_ids})
        keep_indices.update({k: None for k in map_special_tokens(tok_json.get('post_processor'))})
        
    keep_indices = indices_required_for_merges(keep_indices, tok_json['model']['vocab'], tok_json['model']['merges'])
    
    if keep_token_order: 
        keep_indices = sorted(keep_indices)

    # build mapping from old to new id
    mapping = {old: new for new, old in enumerate(keep_indices)}

    # update tokenizer info
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = remove_unused_merges(tok_json['model']['merges'], tok_json['model']['vocab'])
    
    special_tokens_order = [t['id'] for t in tok_json['added_tokens']]
    assert special_tokens_order==sorted(special_tokens_order)
    
    tok_json['added_tokens'] = sorted([{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping], key=lambda t: t['id'])
    map_special_tokens(tok_json.get('post_processor'), mapping)
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))
    return mapping, keep_indices


def shrink_model_embeddings(
    model: PreTrainedModel,
    keep_indices: OrderedDict[int, None],
    mapping: Dict[int, int] # key: old token id, value: new token id
):
    with torch.no_grad():
        # copy input embeddings to lm head
        model.lm_head.weight = nn.Parameter(model.get_input_embeddings().weight.clone())
        print(f"✓ Model output embeddings weight copied from input embeddings")
        
        # copy embeddings to keep
        row_select = torch.tensor(list(keep_indices))
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select.to(model.get_input_embeddings().weight.data.device))
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select.to(model.get_output_embeddings().weight.data.device))
        model.resize_token_embeddings(len(keep_indices))
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))


def apply_custom_head(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    keep_token_ids: List = [], 
    keep_tokens: List = [], 
    remove_token_ids: List = [], 
    keep_model_tokens: bool = True, 
    keep_special_tokens: bool = True, 
    keep_normalizer: bool = False, 
    keep_token_order: bool= True,
    fmt_opts: dict = None,
):
    if not keep_normalizer: 
        remove_tokenizer_normalizer(tokenizer)
    from collections import OrderedDict  # use as OrderedSet
    keep_indices = OrderedDict()
    keep_indices.update({k: None for k in keep_token_ids})
    keep_indices.update({tokenizer.vocab[t]: None for t in keep_tokens})
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
                    "output": [[9, 8], [7, 6]],
                },
            ],
        }
        
        formatted_point = format_prompt_messages(datapoint=sample_datapoint, tokenizer=tokenizer, input_start = fmt_opts["input_start"], input_end = fmt_opts["input_end"], output_end = fmt_opts["output_end"], preprompt = fmt_opts["preprompt"])
        tokens = tokenizer(formatted_point)['input_ids']
        keep_indices.update({k: None for k in tokens})
    except Exception as e:
        print(f"Failed to apply custom head: {e}")
    if keep_model_tokens:
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update({k: None for k in (v if isinstance(v, list) else [v])})
    keep_indices.pop(None, None)
    for idx in remove_token_ids: 
        keep_indices.pop(idx, None)
    mapping, keep_indices = shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens, keep_token_order)
    shrink_model_embeddings(model, keep_indices, mapping=mapping)
    return mapping
import os, glob, json, time, random, copy, gc, datetime, traceback, sys
from wasabi import msg
from omegaconf import DictConfig, OmegaConf
import numpy as np

from trl import SFTTrainer, SFTConfig
from transformers import GenerationConfig, TrainingArguments, EarlyStoppingCallback
import torch
from typing import List, Union, Literal, Any, TypedDict, Callable, Optional
from datasets import load_dataset, Dataset as HFDataset
from pprint import pprint

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import LoraConfig, PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arc import arc_utils, data_transform, data_augmentation
from arc.datatypes import *
from arc.custom_head import *

model_id = "Qwen/Qwen3-0.6B"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
    bnb_4bit_quant_type="nf4",  # Specify the quantization type
    bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
)

model_args = {
    "pretrained_model_name_or_path": model_id,
    "trust_remote_code": True,  # Allow the model to use custom code from the repository
    "quantization_config": bnb_config,  # Apply the 4-bit quantization configuration
    "attn_implementation": "sdpa",  # Use scaled-dot product attention for better performance
    "torch_dtype": torch.float16,  # Set the data type for the model
    "use_cache": False,  # Disable caching to save memory
    "token": None,
    "tie_word_embeddings": False,
    # "device_map": "auto",  # Automatically map the model to available devices
}

# Load tokenizer first so it's available for model optimization
tokenizer_args = {
    "pretrained_model_name_or_path": model_id,
    "token": None,
}
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
tokenizer.bos_token_id = 151643 # Default for Qwen3

base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    **model_args,
).to(device)

keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + ['\n']

# keep_single_char_tokens(model=base_model, tokenizer=tokenizer, keep_tokens=keep_tok, keep_special=True, remove_unk=True)

def tok_to_ids_1(keep_tok: list[str], tokenizer: PreTrainedTokenizer) -> list[int]:
    """
    Convert a list of tokens to their corresponding IDs using the tokenizer.
    """
    ids = []
    for tok in keep_tok:
        tok_tokenized = tokenizer.tokenize(tok)[0]
        if tok != tok_tokenized:
            print(f"Warning: {tok} is not the same as {tok_tokenized}")
        tok_id = tokenizer.convert_tokens_to_ids(tok_tokenized)
        if tok_id != tokenizer.unk_token_id:
            ids.append(tok_id)
    return ids

def tok_to_ids_2(keep_tok: list[str], tokenizer: PreTrainedTokenizer) -> list[int]:
    ids = []
    for tok in keep_tok:
        tok_id = tokenizer.encode(tok, add_special_tokens=False)[0]
        if tok_id != tokenizer.unk_token_id:
            ids.append(tok_id)
    return ids

def tok_to_ids_3(keep_tok: list[str], tokenizer: PreTrainedTokenizer) -> list[int]:
    ids = []
    for tok in keep_tok:
        tok_id = tokenizer(tok, add_special_tokens=False).input_ids[0]
        if tok_id != tokenizer.unk_token_id:
            ids.append(tok_id)
    return ids

def tok_to_ids_4(keep_tok: list[str], tokenizer: PreTrainedTokenizer) -> list[int]:
    ids = []
    for tok in keep_tok:
        tokenized = tokenizer.tokenize(tok)[0]
        tok_id = tokenizer.get_vocab()[tokenized]
        if tok_id != tokenizer.unk_token_id:
            ids.append(tok_id)
    return ids

print("keep_tok", keep_tok)
ids1 = sorted(tok_to_ids_1(keep_tok, tokenizer))
print("ids1", ids1)
ids2 = sorted(tok_to_ids_2(keep_tok, tokenizer))
print("ids2", ids2)
ids3 = sorted(tok_to_ids_3(keep_tok, tokenizer))
print("ids3", ids3)
ids4 = sorted(tok_to_ids_4(keep_tok, tokenizer))
print("ids4", ids4)
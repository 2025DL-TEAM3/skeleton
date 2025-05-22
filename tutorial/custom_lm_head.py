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
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arc import arc_utils, data_transform, data_augmentation
from arc.datatypes import *
from arc.custom_head import *

model_id = "Qwen/Qwen3-4B"
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

input_emb = base_model.get_input_embeddings().weight
lm_head = base_model.lm_head.weight
output_emb = base_model.get_output_embeddings().weight

print(input_emb.data_ptr() == lm_head.data_ptr())
print(input_emb.data_ptr() == output_emb.data_ptr())
print(lm_head.data_ptr() == output_emb.data_ptr())

print(torch.allclose(input_emb, lm_head))
print(torch.allclose(input_emb, output_emb))
print(torch.allclose(lm_head, output_emb))

apply_custom_head(base_model, tokenizer)


input_emb = base_model.get_input_embeddings().weight
lm_head = base_model.lm_head.weight
output_emb = base_model.get_output_embeddings().weight

print(input_emb.data_ptr() == lm_head.data_ptr())
print(input_emb.data_ptr() == output_emb.data_ptr())
print(lm_head.data_ptr() == output_emb.data_ptr())

print(torch.allclose(input_emb, lm_head))
print(torch.allclose(input_emb, output_emb))
print(torch.allclose(lm_head, output_emb))
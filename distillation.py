from transformers import AutoTokenizer, PreTrainedTokenizerBase

from myarc.arc_dataset import build_hf_dataset
from myarc import ARCSolver

from pprint import pprint
hf_dataset = build_hf_dataset(
    # dataset_path="../dataset",
    dataset_path=None,
    reasoning_task_path="reasoning_summary_results_backup",
    num_train_examples_per_normal_task=4,
)

hf_dataset_splitted = hf_dataset.train_test_split(test_size=0.1)

solver = ARCSolver(
    checkpoint_save_path="artifacts/distillation/test",
)
# solver.train(
#     train_dataset=hf_dataset_splitted["train"],
#     eval_dataset=hf_dataset_splitted["test"],
#     num_epochs=4,
#     batch_size=1,
# )


# tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# print("hi")
# for i, example in enumerate(hf_dataset):
#     print(f"---------------------- {i} --------------------")
#     prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
#     prompt_completion = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False)
#     completion = prompt_completion[len(prompt):]
#     print(prompt)
#     print("=" * 20)
#     print(completion)
#     print()
    
#     processed_prompt = tokenizer(prompt, add_special_tokens=False)
#     processed = tokenizer(prompt + completion, add_special_tokens=False)
#     print(processed["input_ids"])
#     break

# from transformers import pipelines

solver.prepare_evaluation(
    checkpoint_name="final",
    enable_ttt=False,
    enable_thinking=True,
)

examples = [
    {
        "input": [
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                3,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                4
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ]
        ],
        "output": [
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                3,
                3,
                3,
                3,
                3,
                3
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                4,
                4,
                4,
                4,
                4,
                4
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ]
        ]
    },
    {
        "input": [
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                3,
                5,
                5,
                5,
                5,
                5
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                6,
                5,
                5,
                5,
                5,
                5
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ]
        ],
        "output": [
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                3,
                3,
                3,
                3,
                3,
                3
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                6,
                6,
                6,
                6,
                6,
                6
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ],
            [
                5,
                5,
                5,
                5,
                5,
                5
            ]
        ]
    },
    {
        "input": [
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                3,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                4,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ]
        ],
        "output": [
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                3,
                3,
                3,
                3,
                3,
                3
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            [
                4,
                4,
                4,
                4,
                4,
                4
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1
            ]
        ]
    }
]

questions_input = [
    [
        7,
        7,
        7,
        6,
        7,
        7,
        7,
        3,
        7
    ],
    [
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    ],
    [
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    ],
    [
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    ],
    [
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    ]
]

solver.predict(
    examples=examples,
    questions_input=questions_input,
)

# from transformers import AutoModelForCausalLM
# from transformers import BitsAndBytesConfig
# import torch

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Enable 4-bit quantization
#     bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
#     bnb_4bit_quant_type="nf4",  # Specify the quantization type
#     bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
# )

# model_args = {
#     "pretrained_model_name_or_path": "Qwen/Qwen3-4B",
#     "trust_remote_code": True,  # Allow the model to use custom code from the repository
#     "quantization_config": bnb_config,  # Apply the 4-bit quantization configuration
#     "attn_implementation": "sdpa",  # Use scaled-dot product attention for better performance
#     "torch_dtype": torch.float16,  # Set the data type for the model
#     "use_cache": False,  # Disable caching to save memory
#     "token": None,
#     "device_map": "auto",  # Automatically map the model to available devices
# }
        
# model = AutoModelForCausalLM.from_pretrained(**model_args)
# model.eval()

# from myarc import arc_utils
# input_messages = arc_utils.format_prompt_messages({
#     "train": examples,
#     "test": [{
#         "input": questions_input,
#         "output": None,
#     }],
# })
# input_text = tokenizer.apply_chat_template(
#     input_messages, 
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True,
# )

# model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
# output_ids = model.generate(**model_inputs, max_new_tokens=32768)

# output_ids_ = output_ids[0][len(model_inputs.input_ids[0]):].tolist()

# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids_[::-1].index(151668)
# except ValueError:
#     index = 0

# # 
# thinking_content = tokenizer.decode(output_ids_[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids_[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:\n", thinking_content)
# print()
# print("content:\n", content)
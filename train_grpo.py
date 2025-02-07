import os
from dataclasses import dataclass, field

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
from trl import (
    GRPOConfig, 
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
)

from prompts import CREATE_DATASET
from rewards import REWARD_FUNCS
from evals import EVALS


@dataclass
class CustomScriptArguments(ScriptArguments):
    dataset_path: str = field(
        default=None,
        metadata={"help": "For offline purpose, path to the cached dataset."}
    )
    num_train_samples: int = field(
        default=-1,
        metadata={"help": "For debug purpose, -1 means the whole training set."}
    )
    num_test_samples: int = field(
        default=-1,
        metadata={"help": "For debug purpose, -1 means the whole test set."}
    )


def main():
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": script_args.gradient_checkpointing_use_reentrant}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset and process
    if script_args.dataset_path:
        dataset = load_from_disk(script_args.dataset_path)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = dataset.map(CREATE_DATASET[script_args.dataset_name], fn_kwargs={'tokenizer': tokenizer})

    # Evaluate before training
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if training_args.do_predict:
        outfile = f"{training_args.output_dir}/predict.jsonl"
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=script_args.max_completion_length,
            n=1,
        )
        EVALS[script_args.dataset_name](
            model_path=model_args.model_name_or_path, 
            num_gpus=len(available_gpus),
            sampling_params=sampling_params,
            seed=training_args.seed,
            dataset=(
                dataset["test"] if script_args.num_test_samples == -1 
                else dataset['test'][:script_args.num_test_samples]
            ),
            outfile=outfile,
        )

    # Load model
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True,
        device_map="auto", 
        **model_kwargs
    )

    # Train
    if training_args.do_train:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=REWARD_FUNCS[script_args.dataset_name],
            args=training_args,
            train_dataset=(
                dataset['train'] if script_args.num_train_samples == -1 
                else dataset['train'][:script_args.num_train_samples]
            ),
        )
        trainer.train()

        # Evaluate after training
    

if __name__=="__main__":
    main()
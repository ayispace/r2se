#!/usr/bin/env python3
"""R²-Searcher SFT LoRA engineering smoke/full run, adapted for one V100 GPU.

Engineering adaptation (not an exact reproduction of the original SFT setup):
- Qwen2.5-3B-Instruct FP16 instead of original Qwen3-4B BF16 script.
- Supervise response tokens only; add EOS as final supervised target.
- Seeded 10% internal validation split, 2-step smoke, no external logging.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 666
MAX_LEN = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA not detected'
    assert torch.cuda.device_count() == 1, 'Expose only one training GPU (e.g. CUDA_VISIBLE_DEVICES=0)'

    data_path = ROOT / 'sft' / 'sft_RB_adv4.jsonl'
    model_path = ROOT / 'models' / 'Qwen2.5-3B-Instruct'
    output_path = ROOT / 'outputs' / ('sft_smoke_v100' if args.mode == 'smoke' else 'sft_full_v100')
    assert data_path.is_file(), f'SFT file not found: {data_path}'
    assert (model_path / 'config.json').is_file(), f'Model not found: {model_path}'
    output_path.mkdir(parents=True, exist_ok=True)

    with data_path.open('r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    assert samples and all(isinstance(x.get('input'), str) and isinstance(x.get('output'), str) for x in samples), 'Malformed input/output fields'
    print('SFT_COUNT=', len(samples), flush=True)
    print('TYPE_DISTRIBUTION=', dict(Counter(x.get('type', 'unknown') for x in samples)), flush=True)

    rng = random.Random(SEED)
    rng.shuffle(samples)
    validation_count = round(len(samples) * 0.10)
    validation_rows = samples[:validation_count]
    train_rows = samples[validation_count:]
    if args.mode == 'smoke':
        train_rows = train_rows[:32]
        validation_rows = validation_rows[:8]
    print(f'TRAIN_ROWS={len(train_rows)} VAL_ROWS={len(validation_rows)}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    def preprocess(example):
        prompt = tokenizer(example['input'], add_special_tokens=False)['input_ids']
        completion = tokenizer(example['output'], add_special_tokens=False)['input_ids']
        assert completion, 'Empty completion'
        if tokenizer.eos_token_id is not None and completion[-1] != tokenizer.eos_token_id:
            completion = completion + [tokenizer.eos_token_id]
        sequence = prompt + completion
        if len(sequence) > MAX_LEN:
            raise ValueError(f'Token length {len(sequence)} > {MAX_LEN}; stop instead of truncating supervision')
        return {
            'input_ids': sequence,
            'attention_mask': [1] * len(sequence),
            'labels': [-100] * len(prompt) + completion,
        }

    train_dataset = Dataset.from_list([preprocess(x) for x in train_rows])
    eval_dataset = Dataset.from_list([preprocess(x) for x in validation_rows])
    print('TOKENIZATION_OK', flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias='none',
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=5,
        max_steps=2 if args.mode == 'smoke' else -1,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1 if args.mode == 'smoke' else 10,
        eval_strategy='no' if args.mode == 'smoke' else 'epoch',
        save_strategy='no' if args.mode == 'smoke' else 'epoch',
        save_total_limit=2,
        report_to='none',
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim='adamw_torch',
        max_grad_norm=1.0,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100
        ),
    )
    print('START_SFT', args.mode, flush=True)
    result = trainer.train()
    print('TRAIN_FINISHED', result.metrics, flush=True)

    adapter_path = output_path / 'final_adapter'
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    assert (adapter_path / 'adapter_config.json').is_file(), 'Adapter was not saved'
    print('SFT_ADAPTER_SAVE_OK:', adapter_path, flush=True)


if __name__ == '__main__':
    main()

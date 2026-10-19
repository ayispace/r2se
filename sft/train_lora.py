# train_lora.py
import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb

def load_sft_data(jsonl_path):
    """Load your SFT data from JSONL"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Combine input + output as full prompt-response
                text = item["input"] + item["output"]
                data.append({"text": text})
    with open('./sft_train_adv_2.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Combine input + output as full prompt-response
                text = item["input"] + item["output"]
                data.append({"text": text})
    with open('./sft_train_adv_3.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Combine input + output as full prompt-response
                text = item["input"] + item["output"]
                data.append({"text": text})
    with open('./sft_train_adv_4.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Combine input + output as full prompt-response
                text = item["input"] + item["output"]
                data.append({"text": text})
    return Dataset.from_list(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./sft_train_2.jsonl")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output_dir", type=str, default="./sft_model")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="fact-searcher-sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    args = parser.parse_args()

    # === 初始化 W&B ===
    wandb.init(
        project=args.wandb_project,
        config=vars(args)  # 自动记录超参数
    )

    # === 1. Load tokenizer and model ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=args.load_in_4bit,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # === 2. Apply LoRA ===
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Should show ~1-2% of full params

    # === 3. Load dataset ===
    raw_dataset = load_sft_data(args.data_path)

    # === Tokenization ===
    def tokenize_function(examples):
        # Concatenate input + output
        texts = [inp + out for inp, out in zip(examples["input"], examples["output"])]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,  # dynamic padding via DataCollator
        )
        return tokenized

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
    )

    # === Data Collator ===
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM
    )

    # === Training Args ===
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-4,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="wandb",
        gradient_checkpointing=True,
        remove_unused_columns=True,
        dataloader_num_workers=4,
        optim="paged_adamw_32bit" if args.load_in_4bit else "adamw_torch",
    )

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,  # for logging & save
    )

    # === Train ===
    trainer.train()

    # === Save LoRA ===
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    wandb.finish()
    print(f"✅ LoRA model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
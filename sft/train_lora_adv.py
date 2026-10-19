import json
import pandas as pd
import torch
from datasets import Dataset
# from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab

            
            
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 3072 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        example['input'],
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response

# model_dir = snapshot_download("qwen/Qwen3-4B", cache_dir="./", revision="master")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

train_jsonl_new_path = "sft_RB_adv4.jsonl"
test_jsonl_new_path = "sft_RB_adv4.jsonl"

train_df = pd.read_json(train_jsonl_new_path, lines=True)
# train_1_df = pd.read_json("sft_train_adv2.jsonl", lines=True)
# train_2_df = pd.read_json("sft_train_adv2_2.jsonl", lines=True)
# train_3_df = pd.read_json("sft_train_adv2_3.jsonl", lines=True)
# train_4_df = pd.read_json("sft_train_adv2_4.jsonl", lines=True)
# combined_df = pd.concat([train_df, train_1_df, train_2_df, train_3_df, train_4_df], ignore_index=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
train_dataset = train_dataset.shuffle(seed=666)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16, 
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./sft_model/RB_merge_4B",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=5,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(project="Qwen-fintune", experiment_name="Qwen2.5-7B-merge")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

test_df = pd.read_json(test_jsonl_new_path, lines=True)[:1]

test_text_list = []
for index, row in test_df.iterrows():
    input_value = row['input']
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))
    
swanlab.log({"Prediction": test_text_list})
swanlab.finish()

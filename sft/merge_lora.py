from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base_model, "./sft_model/RB_merge_4B/checkpoint-505")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merge/Qwen3-4B-RB-merge")
tokenizer.save_pretrained("./merge/Qwen3-4B-RB-merge")
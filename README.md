# R2-Searcher
This is the implementation code of R$^2$-Searcher.

## Quick Start

### Installation

#### RL Environment
```bash
conda create -n treegrpo python=3.12.9
conda activate treegrpo
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0  
pip install vllm==0.8.5.post1
# verl
pip install -e .
# flash attention 2
pip install flash-attn --no-build-isolation
pip install swanlab
```

### Retriever Environment
```bash
conda create -n retriever python=3.10.13
conda activate retriever
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0  
pip install transformers datasets pyserini
pip install faiss-gpu==1.7.3
pip install uvicorn fastapi
```


### Dataset

Download the wiki dump indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

Process the single-hop QA and multi-hop QA datasets into parquet.
```bash
bash scripts/data_process/data_process_multihop.sh
```

### RL Training

Launch a local retrieval server for single-hop QA and multi-hop QA,
```bash
conda activate retriever
bash retrieval_launch.sh
```
Then you can start SFT training by using:
```bash
bash sft/train_lora.sh
```
The training data is also included in the sft file. Then you should merge the fine-tuned model for further RL training:
```bash
bash sft/merge_lora.sh
```
And finally you can start RL training:
```bash
bash train.sh
```


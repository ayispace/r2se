#!/bin/bash
#SBATCH --job-name=launch             # 作业名称-请修改为自己任务名字 
#SBATCH --cpus-per-task=4             # 每个任务使用的CPU核心数
#SBATCH --output=/home/szhang844/autorefine/AutoRefine/sft/retriever_data_pre_log.txt        # 标准输出文件名 (%j 表示作业ID)-请修改为自己路径
#SBATCH --error=/home/szhang844/autorefine/AutoRefine/sft/retriever_data_pre_error.txt          # 标准错误文件名-请修改为自己路径
#SBATCH --mem=200G                      # 申请100GB内存
#SBATCH --time=24:00:00               # 运行时间限制，格式为hh:mm:ss

file_path=./data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=intfloat/e5-base-v2

CUDA_VISIBLE_DEVICES="0,3" python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever

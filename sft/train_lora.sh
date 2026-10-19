#!/bin/bash
#SBATCH --job-name=sft            
#SBATCH --cpus-per-task=4 
#SBATCH --mem=200G
#SBATCH --time=12:00:00

CUDA_VISIBLE_DEVICES="1,2,3" python train_lora_adv.py

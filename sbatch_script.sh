#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH -n 1

# not sure; might need something; full path
# source activate acl2020
# source /h/johnchen/anaconda3/bin/activate acl2020

python --dataset ring --cuda --train_iterations 800 --batch_size 16 --sampling_method uncertainty --log_name bs_16 --out_path dontcare --torch_manual_seed 88888



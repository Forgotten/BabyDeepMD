#!/usr/bin/env bash

#SBATCH --partition=batch_default
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00

source ~/.bashrc

module load anaconda
module load cuda 

conda activate tf-2.0-gpu

CUDA_VISIBLE_DEVICES=0 python ../../src/2BodyForcesDist.py Np20_Per_mu_1.json
#python sca_bwd_10h_circle_data.py 


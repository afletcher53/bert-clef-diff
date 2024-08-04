#!/bin/bash
#SBATCH --comment=cdt-transcribe
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<user>@sheffield.ac.uk
#SBATCH --output=log/%j.%x.out
#SBATCH --error=log/%j.%x.err

# load modules
module load Anaconda3/2022.10
module load cuDNN/8.7.0.84-CUDA-11.8.0

# init env
source .venv/bin/activate
PATH=$(pwd)/bin:$PATH

# begin process
python main.py

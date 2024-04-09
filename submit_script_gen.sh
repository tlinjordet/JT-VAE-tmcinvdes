#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=sm3090el8
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --time=16:00:00
#SBATCH --mem=10G     # 10 GB RAM per node
#SBATCH --gres=gpu:RTX3090:1  # Allocate 1 GPU
#SBATCH --output=gen_outfiles/R-%x.%j.out
#SBATCH --error=gen_outfiles/R-%x.%j.err

# Source conda env 
source ~/miniconda3/etc/profile.d/conda.sh # Or path to where your conda is
conda activate vae_py37

# default values
#MEM=2

# read options
while getopts ":m:v:n:c:" OPTION; do
    case $OPTION in
    m)
        model=${OPTARG};;
    v)
        vocab=${OPTARG};;
    n)
        nsample=${OPTARG};;
    c)
        custom_label=${OPTARG};;
    *)  
        echo "Incorrect options provided"
        exit 1
        ;;
    esac
done

echo "Run sampling from model: $model"
echo "Vocab : $vocab"

model_name=${model%/*}

python fast_molvae/sample.py --nsample $nsample --vocab $vocab --model $model --output_file samples/sample_${model_name}_${custom_label}.txt


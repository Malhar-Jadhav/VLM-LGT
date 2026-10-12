#!/bin/bash
#SBATCH -J vlm300
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/vlm300_%j.out

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate ~/scratch/envs/vlm
export HF_HOME=~/scratch/hf_cache
cd ~/scratch/VLM-LGT

python make_gqa_smoke.py --limit 300
python -m data.approach2_verifier_reranking.build_failure_pool --dataset gqa --size 2b --k 4 --limit 300
python -m models.run_verifier_reranking \
  --pool data/approach2_verifier_reranking/failure_pool/gqa_pool.jsonl \
  --verifier-size 2b --limit 300 --output-dir results/approach2/gqa_300

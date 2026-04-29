#!/bin/bash
#SBATCH --job-name=gemma_sft_500
#SBATCH --account=project_2019025
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e

echo "🚀 Starting Gemma SFT run on $SLURMD_NODENAME"

# ------------------------
# ENV SETUP
# ------------------------
module purge
module load pytorch

export HF_HOME="/scratch/project_2019025/hpc-llm-pipeline/hf_cache"
mkdir -p $HF_HOME logs

export HF_TOKEN=${HF_TOKEN}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

export TOKENIZERS_PARALLELISM=false
export ACCELERATE_MIXED_PRECISION=no
export HF_HUB_DISABLE_TELEMETRY=1

# ------------------------
# PROJECT DIR
# ------------------------
cd /scratch/project_2019025/hpc-llm-pipeline

# ------------------------
# ENV
# ------------------------
ENV_NAME="v100_compute_env"

if [ ! -d "$ENV_NAME" ]; then
  echo "📦 Creating env..."
  python3 -m venv --system-site-packages $ENV_NAME
fi

source $ENV_NAME/bin/activate

# ------------------------
# DEPENDENCIES (optional optimization)
# ------------------------
pip install -q -r requirements.txt

# ------------------------
# SAFETY CHECK
# ------------------------
test -f qml_finetune_dataset_v4.jsonl || {
  echo "❌ Dataset missing!"
  exit 1
}

# ------------------------
# RUN TRAINING
# ------------------------
echo "🔥 Launching training..."

python3 src/train_scholar_dryrun.py

echo "✅ Done"

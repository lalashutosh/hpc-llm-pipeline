#!/bin/bash
#SBATCH --job-name=gemma12B_dryrun
#SBATCH --account=project_2019025
# CHANGED: Moving to the main GPU queue to bypass the 15-minute kill switch
#SBATCH --partition=gpu
# CHANGED: 45 minutes gives us plenty of time to download the 12B weights
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
# CHANGED: Requesting 1x V100 (32GB VRAM).
#SBATCH --gres=gpu:v100:1
#SBATCH --output=dryrun_%j.out
#SBATCH --error=dryrun_%j.err
# Fail fast: Instantly kill the job if any command fails
set -e

echo "🚀 Starting Gemma 2B Dry Run on $SLURMD_NODENAME..."

# 1. Clean the slate and load CSC's optimized GPU wrappers
module purge
module load pytorch
export ACCELERATE_MIXED_PRECISION=fp16
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# 2. Go to the correct new project folder
cd /scratch/project_2019025/hpc-llm-pipeline

# 3. Create a strictly isolated environment for the GPU nodes
ENV_NAME="v100_compute_env"

if [ ! -d "$ENV_NAME" ]; then
  echo "📦 Building isolated GPU compute environment..."
  # --system-site-packages inherits the optimized PyTorch from the module
  python3 -m venv --system-site-packages $ENV_NAME
fi

# 4. Activate the isolated environment
source $ENV_NAME/bin/activate

# 5. Install the strictly version-locked dependencies
echo "📥 Installing locked dependencies from requirements.txt..."
pip install --quiet -r requirements.txt

# 6. Inject the Hugging Face Token (REPLACE THIS WITH YOUR REAL TOKEN)
export HF_HOME="/scratch/project_2019025/hpc-llm-pipeline/hf_cache"
export HF_TOKEN=$(cat ~/.hf_token)
# 7. Ignite the Dry Run
echo "🔥 Igniting train_scholar_dryrun.py..."
python3 train_scholar_dryrun.py

echo "✅ Job Complete!"

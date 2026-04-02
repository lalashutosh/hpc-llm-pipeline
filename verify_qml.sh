#!/bin/bash
#SBATCH --job-name=gemma_verify
#SBATCH --account=project_2017556
#SBATCH --partition=gpu
#SBATCH --time=00:30:00          # 30 mins is plenty for a 5-step dry run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Enough to handle data mapping
#SBATCH --mem=64G                # Sufficient for loading 12B weights
#SBATCH --gres=gpu:v100:1        # Start with 1 GPU to minimize queue wait

# 1. Environment Setup
module load pytorch              # Loads underlying CUDA/CUDNN
source /scratch/project_2017556/quantum-slm/qml_env/bin/activate

# 2. Paths & Cache
export HF_HOME="/scratch/project_2017556/quantum-slm/hf_cache"
mkdir -p $HF_HOME

# 3. V100 Memory Optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 4. Run the validation script
# Ensure your Python script has MAX_STEPS=5 and the fixed format_gemma function!
python3 train_scholar_dryrun.py

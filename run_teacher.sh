#!/bin/bash
#SBATCH --job-name=qml_teacher
#SBATCH --account=project_2017556
#SBATCH --partition=gpu
#SBATCH --time=02:00:00          # Allow 2 hours for downloading the model + generation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # More CPUs to feed the GPUs
#SBATCH --mem=128G               # Lots of system RAM for loading weights
#SBATCH --gres=gpu:v100:4        # CRITICAL: Request 4 Tesla V100 GPUs

# 1. Load the modules
module load tykky
module load pytorch

# 2. Activate your virtual environment
source /scratch/project_2017556/quantum-slm/qml_env/bin/activate

# 3. CRITICAL: Redirect HuggingFace cache to your scratch drive so you don't run out of disk space
export HF_HOME="/scratch/project_2017556/quantum-slm/hf_cache"
mkdir -p $HF_HOME

# Prevent PyTorch memory fragmentation on V100s
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 4. Install vLLM (if you haven't already in this env)
pip install vllm pydantic

# 5. Run the script
python3 teacher_pipeline.py

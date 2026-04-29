#!/bin/bash
#SBATCH --job-name=qml_teacher
#SBATCH --account=project_2019025
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100:2
#SBATCH --output=teacher_%j.out
#SBATCH --error=teacher_%j.err

echo "🚀 Starting QML Teacher Pipeline on $SLURMD_NODENAME"

# 1. Clean environment for reproducibility
module purge
module load pytorch

# 2. Go to project directory
cd /scratch/project_2017556/quantum-slm

# 3. Activate venv
source qml_env/bin/activate

# 4. HuggingFace cache on scratch (IMPORTANT)
export HF_HOME="/scratch/project_2017556/quantum-slm/hf_cache"
mkdir -p $HF_HOME

# 5. Prevent CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 6. (Optional but recommended) sanity check GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"

# 7. Run pipeline
echo "🔥 Running teacher_pipeline.py"
python3 teacher_pipeline.py

echo "✅ Job finished successfully"

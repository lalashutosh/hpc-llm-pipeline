#!/bin/bash
#SBATCH --job-name=qml_cluster
#SBATCH --account=project_2017556
#SBATCH --partition=gpu          
#SBATCH --time=00:30:00          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                
#SBATCH --gres=gpu:v100:1        

# Load the base module
module load pytorch

# Activate your custom environment
source /scratch/project_2017556/quantum-slm/qml_env/bin/activate

# Run the script
python3 cluster_and_select.py

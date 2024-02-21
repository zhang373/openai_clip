#!/bin/bash -x
#SBATCH -p czhangcn_rent
#SBATCH -J openclip-8gpu-q1m-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH -o oldclip-8gpu-q1m-run_0207_1.out   # 作业运行log输出文件
#SBATCH -e oldclip-8gpu-q1m-run_0207_1.err   # 作业错误信息log输出文件

# store the starting time
start=`date +%s`

nvidia-smi
module load anaconda3
module load cuda/12.2
module load slurm


source activate monodepth2


echo "We are going to run the main file"
python tune_clip.py

# store the ending time
end=`date +%s`
runtime=$((end-start))
echo "Finished! Total time: $runtime"

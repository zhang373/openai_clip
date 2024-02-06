#!/bin/bash -x
#SBATCH -p czhangcn_rent
#SBATCH -J openclip-8gpu-q1m-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH -o openclip-8gpu-q1m-test.out   # 作业运行log输出文件
#SBATCH -e openclip-8gpu-q1m-test.err   # 作业错误信息log输出文件


cp -r /home/jovyan/.conda/envs/ /hpc2hdd/home/wenshuozhang/.conda/

#!/usr/bin/bash
# Parameters
#SBATCH --error=outputs/slurm/%j_%t_log.err
#SBATCH --job-name=train
#SBATCH --partition=3090
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=outputs/slurm/%j_%t_log.out
#SBATCH --exclude=

export FFCV_DEFAULT_CACHE_PROCESS=0
# mkdir -p /raid/local_scratch/jxw30-hxc19/ffcv/
# rsync /jmain02/home/J2AD011/hxc19/jxw30-hxc19/data/ffcv/IN1K_smart_500.ffcv /raid/local_scratch/jxw30-hxc19/ffcv/IN1K_smart_500.ffcv
# train_path=/raid/local_scratch/jxw30-hxc19/ffcv/IN1K_smart_500.ffcv
ARGS={@:1}

apptainer exec ../outputs/APPTAINER/torch2.sif bash -c "pip install -r requirements.txt && pip install -e ../ffcv && torchrun --nproc_per_node=4 $ARGS"

# docker run --gpus all --rm --mount type=bind,source=$(pwd),target=/workspace -it  ghcr.io/erow/aisurrey-docker:main /bin/bash -c " pip install -r requirements.txt&& torchrun --nproc_per_node=4 $ARGS"
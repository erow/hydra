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

export ARGS=${@:2}
if [ -z "$master" ]; then
# master
    
    export master=`hostname`
    export NPROC_PER_NODE=4
    export GPUS=$1
    export RANK=0
    # launch slaver
    for ((i = 1; i <= $GPUS-1; i++ )); do 
        sbatch storchrun.sh ${i} $ARGS; 
    done
    echo torchrun --nnodes=${GPUS} --node-rank=0 --master-addr $master --nproc_per_node=$NPROC_PER_NODE $ARGS
    # nvidia-smi

    apptainer exec ../outputs/APPTAINER/torch2.sif bash train.sh
else
# slave
    export RANK=$1
    # rsync --update -av /jmain02/home/J2AD011/hxc19/jxw30-hxc19/data/ffcv /raid/local_scratch/$USER
    echo torchrun --nnodes ${GPUS} --node-rank ${rank} --master-addr $master --nproc_per_node=8 $ARGS
    # nvidia-smi
    # python main_moco.py --dist-url "tcp://${master-addr}:29532" --multiprocessing-distributed --world-size $GPUS --rank $rank  --moco-m-cos --crop-min=.2 --moco-t 0.05 ~/data/ImageNet -b 1024 --output_dir outputs/pretrain/hydra_moco_conv_r50_e100/

    torchrun --nnodes ${GPUS} --node-rank ${rank} --master-addr $master --nproc_per_node=$NPROC_PER_NODE $ARGS
fi

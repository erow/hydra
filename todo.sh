# define environment variables
export WANDB_PROJECT=hydra
export WANDB_ENTITY=dlib
export OUTDIR=../outputs/coco
export train_path=/mnt/fast/datasets/still/MSCOCO/
export EMBEDFILE=../outputs/coco_embed.pth
export launcher="sbatch -p a100 storchrun.sh 1" # launch jobs on slurm
# export launcher="sbatch -p 2080ti svitrunrun.sh" # evaluate on slurm
# launcher="torchrun --nproc_per_node=8 " # launch jobs on local machine
# export launcher="echo" # dry run

export WANDB_TAGS=coco

# WANDB_NAME=vitb_clip $launcher main_moco.py  -a vit_base -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40  --crop-min=0.4 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --moco-dim=512   --gin MoCo.clip=True  --output_dir $OUTDIR/clip  --embed_file EMBEDFILE $train_path


for beta in 0 1 2 10 100; do
    WANDB_NAME=vitb_hydra_beta${beta} $launcher main_moco.py -a vit_base -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40  --crop-min=0.4 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --moco-dim=512   --gin MoCo.clip=False   --gin MoCo.beta=${beta} --output_dir $OUTDIR/hydra_beta${beta} --embed_file $EMBEDFILE $train_path
done
# define environment variables
export WANDB_PROJECT=hydra
export WANDB_ENTITY=dlib
export OUTDIR=../outputs/
export train_path=/mnt/fast/datasets/still/MSCOCO/
export EMBEDFILE=../outputs/coco_embed.pth
export launcher="sbatch -p a100 storchrun.sh " # launch jobs on slurm
# export launcher="sbatch -p 2080ti svitrunrun.sh" # evaluate on slurm
# launcher="torchrun --nproc_per_node=8 " # launch jobs on local machine
# export launcher="echo" # dry run

export WANDB_TAGS=coco

# WANDB_NAME=vitb_clip $launcher main_moco.py  -a vit_base -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40  --crop-min=0.4 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --moco-dim=512   --gin MoCo.beta=0 MoCo.alpha=0  --output_dir $OUTDIR/coco/clip  --embed_file EMBEDFILE $train_path

# WANDB_NAME=vitb_moco $launcher main_moco.py  -a vit_base -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40  --crop-min=0.4 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --moco-dim=512   --gin MoCo.beta=0 MoCo.alpha=100  --output_dir $OUTDIR/coco/moco  --embed_file EMBEDFILE $train_path

WANDB_NAME=vitb_clip_hydra_moco $launcher main_moco.py  -a vit_base -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40  --crop-min=0.4 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --moco-dim=512   --gin MoCo.beta=1 MoCo.alpha=1  --output_dir $OUTDIR/coco/clip_hydra_moco  --embed_file EMBEDFILE $train_path

WANDB_NAME=vitb_clip_moco $launcher main_moco.py  -a vit_base -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40  --crop-min=0.4 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --moco-dim=512   --gin MoCo.beta=0 MoCo.alpha=1  --output_dir $OUTDIR/coco/clip_moco  --embed_file EMBEDFILE $train_path


WANDB_NAME=vitb_clip_hydra $launcher main_moco.py  -a vit_base -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40  --crop-min=0.4 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --moco-dim=512   --gin MoCo.beta=1 MoCo.alpha=0  --output_dir $OUTDIR/coco/clip_hydra  --embed_file EMBEDFILE $train_path

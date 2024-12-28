# define environment variables
export WANDB_PROJECT=hydra
export WANDB_ENTITY=dlib
export OUTDIR=../outputs
export train_path=../outputs/data/IN1K_smart_500.ffcv
# export launcher="sbatch -p a100 storchrun.sh 1" # launch jobs on slurm
# launcher="torchrun --nproc_per_node=8 " # launch jobs on local machine
export launcher="echo" # dry run
################# component design: vit_tiny-IN1K #################


# baseline
WANDB_NAME=hydra_vitt_baseline $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=100 MoCo.norm=\'ln-none\'  --output_dir $OUTDIR/design/hydra_vitt_baseline   --data_set ffcv $train_path

######## gate vector without learning ##########
WANDB_NAME=hydra_vitt_gate $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=1 MoCo.learnable=False  --output_dir $OUTDIR/design/hydra_vitt_gate   --data_set ffcv $train_path

######## separated predictor ##########
WANDB_NAME=hydra_vitt_sep $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=1 MoCo.sep=True  --output_dir $OUTDIR/design/hydra_vitt_sep   --data_set ffcv $train_path

######## batch size ##########
## we study the effect of the batch size on the performance of the model
export WANDB_TAGS="vitt,bs"
for bs in 2048 4096 8192; do

    WANDB_NAME=hydra_bs${bs} $launcher main_moco.py  -a vit_tiny -b ${bs}   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=100   --output_dir $OUTDIR/design/hydra_bs${bs}   --data_set ffcv $train_path

done

######## data augmentation ##########
## we study the effect of the data augmentation on the performance of the model
export WANDB_TAGS="vitt,aug"
for aug in  "simple"; do

    WANDB_NAME=hydra_${aug} $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=100 --aug=${aug}  --output_dir $OUTDIR/design/hydra_${aug}   --data_set ffcv $train_path

done

######## norm layer ##########
## we study the effect of the normalization layer at the end of the projector and the predictor
export WANDB_TAGS="vitt,norm"
for norm in "bn-none" "bn-ln" "ln-none" "ln-ln" "bn-bn"; do

    WANDB_NAME=hydra_${norm} $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=100 MoCo.norm=\'${norm}\'  --output_dir $OUTDIR/design/hydra_${norm}   --data_set ffcv $train_path

#     WANDB_NAME=moco_${norm} $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=0 MoCo.norm=\'${norm}\'  --output_dir $OUTDIR/moco_${norm}   --data_set ffcv $train_path
done

######## projector ##########
## we study the effect of the projector on the performance of the model
export WANDB_TAGS="vitt,proj"

## linear projector
WANDB_NAME=hydra_proj1 $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=100 MoCo.num_layers=1  --output_dir $OUTDIR/design/hydra_proj1   --data_set ffcv $train_path

## no projection
WANDB_NAME=hydra_proj0 $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-dim=192 --moco-t=.2  --gin MoCo.beta=100 MoCo.num_layers=0 MoCo.norm=\'none-none\' --output_dir $OUTDIR/design/hydra_proj0   --data_set ffcv $train_path


######## beta ##########
## we study the effect of the beta parameter on the performance of the model
export WANDB_TAGS="vitt,beta"
for beta in 0 1 10; do
    WANDB_NAME=hydra_beta${beta} $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=${beta}  --output_dir $OUTDIR/design/hydra_beta${beta}   --data_set ffcv $train_path
done

# 17-12-2024
######## gamma ##########
export WANDB_TAGS="vitt,gamma"
for G in 1 1e-1 1e-2; do
    WANDB_NAME=hydra_g${dim} $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.gamma=${G} MoCo.beta=1 --output_dir $OUTDIR/design/hydra_g${dim}   --data_set ffcv $train_path
done

######## dim ##########
export WANDB_TAGS="vitt,dim"
for dim in 128 512 1024; do
    WANDB_NAME=hydra_d${dim} $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --moco-dim=${dim}  --output_dir $OUTDIR/design/hydra_d${dim}   --data_set ffcv $train_path
done


##### model size ######
WANDB_NAME=hydra_vitt $launcher main_moco.py  -a vit_tiny -b 1024   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=1 MoCo.norm=\'bn-ln\'  --seed=1 --output_dir $OUTDIR/hydra_vitt_s1   --data_set ffcv $train_path

WANDB_NAME=hydra_vits $launcher main_moco.py  -a vit_small -b 4096   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=1 MoCo.grad_checkpointing=True MoCo.norm=\'bn-ln\' --seed=1 --output_dir $OUTDIR/hydra_vits_s1   --data_set ffcv $train_path


WANDB_NAME=hydra_vitb $launcher main_moco.py  -a vit_base -b 4096  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1   --epochs=300 --warmup-epochs=40   --stop-grad-conv1 --moco-m-cos --moco-t=.2  --gin MoCo.beta=1 MoCo.grad_checkpointing=True MoCo.norm=\'bn-ln\'  --seed=1 --output_dir $OUTDIR/hydra_vitb_s1    --data_set ffcv $train_path

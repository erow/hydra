## Hydra MoCo Reference Setups and Models

Here we document the reference commands for pre-training and evaluating various Hydra MoCo models.

### ResNet-50 models

With batch 4096, the training of all ResNet-50 models can fit into 2 nodes with a total of 16 Volta 32G GPUs. 

<details>
<summary>ResNet-50, 100-epoch pre-training.</summary>

Run:
```
torchrun --nproc_per_node=8 main_moco.py \
  --moco-m-cos --crop-min=.2 --moco-t 0.15 \
  -b 2048 [your imagenet-folder with train and val folders]\
  --output_dir [outputs]
```
</details>

<details>
<summary>ResNet-50, 300-epoch pre-training.</summary>

Run:
```
torchrun --nproc_per_node=8 main_moco.py \
  --lr=.3 --epochs=300 \
  --moco-m-cos --crop-min=.2 --moco-t 0.15 \
  -b 4096 [your imagenet-folder with train and val folders]\
  --output_dir [outputs]
```
</details>

<details>
<summary>ResNet-50, 1000-epoch pre-training.</summary>

Run:
```
torchrun --nproc_per_node=8 main_moco.py \
  --lr=.3 --wd=1.5e-6 --epochs=1000 \
  --moco-m=0.996 --moco-m-cos --crop-min=.2 --moco-t 0.15 \
  -b 4096 [your imagenet-folder with train and val folders]\
  --output_dir [outputs]
```
</details>

<details>
<summary>ResNet-50, 1000-epoch pre-training using ffcv.</summary>

Run:
```
torchrun --nproc_per_node=8 main_moco.py \
  --lr=.3 --wd=1.5e-6 --epochs=1000 \
  --moco-m=0.996 --moco-m-cos --crop-min=.2 --moco-t 0.15 \
  --gin MultiviewPipeline.scale="(0.2, 1)" -b 4096 --workers 40 --data_set ffcv [your dataset file] \
  --output_dir [outputs]
```
</details>
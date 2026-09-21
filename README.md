# Backbone extraction

`extract_backbone.py` extracts the feature extractor from every checkpoint in
`../weights/`. It runs on CPU and writes a plain PyTorch state dictionary; it
removes ImageNet classifiers, MoCo projectors, Hydra filters, and predictors.

## Backbone structures

| Checkpoint | Backbone | Removed keys |
| --- | --- | --- |
| `SimLAP_v1_e1000_IN1K_resnet.pth` (alias `HydraV1_e1000_…`) | ResNet-50: `conv1`, `bn1`, `layer1`–`layer4` | `fc.*` |
| `SimLAP_v2_e1000_IN1K_resnet.pth` (alias `HydraV2_e1000_…`) | ResNet-50: `conv1`, `bn1`, `layer1`–`layer4` | none |
| `Hydra_moco_e300_IN1K_resnet.pth` | ResNet-50: `conv1`, `bn1`, `layer1`–`layer4` | `fc.*` projector |
| `HydraV1_resnet.pth`, `rebuttal_dp1.ckpt` | ResNet-50 under `visual.` | `visual.fc.*`, non-visual modules |
| `Hydra_e100_IN1K_vitt.pth` | ViT-tiny: patch embedding, 12 blocks, `norm` | `head.*` |
| `Hydra_e300_IN1K_vits.pth` | ViT-small: patch embedding, 12 blocks, `norm` | `head.*` |
| `Hydra_e300_IN1K_vitb.pth` | ViT-base under `visual.` | `visual.head.*`; drop `projector`/`filter`/`logit_scale` |
| `hydra-moco_bs4k_mae.pth` | ViT-base: patch embedding, 12 blocks, `norm` | `module.base_encoder.head.*` |

Output keys are normalized to model-local names. For example,
`module.base_encoder.blocks.0.attn.qkv.weight` becomes
`blocks.0.attn.qkv.weight`, while `visual.layer1.0.conv1.weight` becomes
`layer1.0.conv1.weight`.

## Extract one checkpoint

Run from the repository root (`src/`):

```bash
python extract_backbone.py \
  ../weights/Hydra_e300_IN1K_vits.pth \
  --output ../results/backbones/Hydra_e300_IN1K_vits_backbone.pth
```

Without `--output`, the default is
`results/backbones/<checkpoint-name>_backbone.pth` relative to the current
directory.

## Extract all checkpoints

```bash
for checkpoint in ../weights/*.pth ../weights/*.ckpt; do
  [ -f "$checkpoint" ] || continue
  python extract_backbone.py "$checkpoint"
done
```

Load an extracted state dictionary into the matching backbone:

```python
import torch

backbone = torch.load(
    "../results/backbones/Hydra_e300_IN1K_vits_backbone.pth",
    map_location="cpu",
)
model.load_state_dict(backbone, strict=True)
```

Use ResNet-50 for the ResNet files, `vit_tiny` for `Hydra_e100_IN1K_vitt.pth`,
`vit_small` for `Hydra_e300_IN1K_vits.pth`, and `vit_base` for
`Hydra_e300_IN1K_vitb.pth` / `hydra-moco_bs4k_mae.pth`.

## SSL representation-comparison model inventory

[`model_manifest.json`](model_manifest.json) is the versioned, declarative
inventory for the frozen-representation comparison. It records each expected
local Hydra checkpoint, the compatible `erow/SSL` Hugging Face comparators,
the pinned Hugging Face revision and LFS hash, architecture, parameter budget,
pretraining metadata, preprocessing, checkpoint layout, and current loading
status. It is metadata only: it never downloads checkpoint weights.

The manifest pins `erow/SSL` to revision
`72d63866af2338351ac142c457c64f73eb1e2490`. A later Hub revision must not be
used without updating both the revision and file hash in the manifest.

### Comparison eligibility

Models may be compared only when all of the following hold:

- their `architecture_id` is identical (ResNet-50, ViT-Small/16, and ViT-Base/16
  remain separate groups);
- both have ImageNet-1k pretraining and 224-pixel inputs;
- both pretraining durations are known and the comparator-to-Hydra epoch ratio
  lies in `[0.75, 1.3333333333]`; and
- both checkpoints have a verified loader before producing metrics.

The approved groups are listed in `approved_comparison_groups`. This currently
allows ResNet-50 1000-epoch comparisons, the explicitly bounded
ResNet-50 300-to-400-epoch comparison, and the ViT-Small 300-epoch comparison.
Unknown budgets, a missing matching architecture, and unverified checkpoint
layouts are explicit exclusions rather than permissive fallbacks.

### Local checkpoint availability

At inventory time `weights/` contains no Hydra checkpoint payloads, so every
local model has `availability: "not_present_locally"` and
`load_status: "unvalidated"`. Supply the intended checkpoint at the recorded
path, calculate its SHA-256, and replace the null `sha256` before evaluation.
Remote comparator layouts are likewise unvalidated because their weights were
not downloaded. Do not interpret a model being in an approved group as proof
that it can be loaded; successful strict backbone loading is a prerequisite for
the later evaluation task.

## Frozen-feature evaluation

`evaluate_frozen.py` is the reusable, declarative evaluation entry point. It
reads `model_manifest.json`, loads one manifest checkpoint, freezes the
backbone, caches normalized 224-pixel embeddings, and fits a closed-form
regularized linear probe. Checkpoint loading is strict for all non-head
parameters; missing local files, unavailable optional datasets, and unverified
checkpoint layouts fail with an actionable error.

After installing `requirements.txt`, a transfer run is:

```bash
python evaluate_frozen.py \
  --model simlap-v1-rn50-e1000 --dataset cifar10 \
  --data-root /path/to/datasets \
  --cache-dir ../results/ssl-representation-comparison/embeddings \
  --output ../results/ssl-representation-comparison/cifar10.json
```

The default protocol evaluates 1, 5, 10, and 25 examples per class with seeds
0, 1, and 2. Override `--shots`, `--seeds`, or `--regularization` for smoke
runs. CIFAR datasets must already be present (the evaluator never downloads
them); Oxford Pets and Flowers must use the layouts documented in
`transfer/README.md`. Remote `erow/SSL` files require the optional
`huggingface_hub` package and `--download`; the manifest revision remains
pinned.

ImageNet-C uses the same frozen representation and a clean ImageNet linear
probe:

```bash
python evaluate_frozen.py --task imagenet-c \
  --model simlap-v1-rn50-e1000 \
  --imagenet-root /path/to/imagenet \
  --imagenet-c-root /path/to/imagenet-c
```

`--imagenet-root` must contain `train/` and `val/`; ImageNet-C must contain
the 15 corruption directories, each with severity directories `1` through
`5` in ImageFolder layout. The output reports clean accuracy, mean corruption
accuracy, relative corruption error, and per-corruption accuracy. This command
does not submit jobs or aggregate reports.

## Slurm launchers and result validation

Remote jobs go through the unified entry `cluster/submit.sh`, which loads
`cluster/isambard.env` (or `cluster/eureka.env`) and runs
`cluster/isambard.sbatch`. Paths are pinned in the env files:

- project: `/lus/lfs1aip2/projects/u6gd/jiantao/SimLAP`
- environment: `/lus/lfs1aip2/projects/u6gd/jiantao/FastSSL/.venv`
- transfer datasets: `DATA_ROOT` (defaults to `/lus/lfs1aip2/projects/u6gd/datasets`)
- licensed ImageNet-1K: `/lus/lfs1aip2/projects/u6gd/datasets/IN1K` (`train/` and class-organized `val/` required)
- public ImageNet-C: `/lus/lfs1aip2/projects/u6gd/datasets/IN1K-C` (15 corruptions × severities `1`–`5`)
- outputs: `/scratch/u6gd/jw02425.u6gd/SimLAP/ssl-representation-comparison/outputs`

Override `IMAGENET_ROOT` / `IMAGENET_C_ROOT` via the environment if needed.
Verify the ImageNet-1K validation split is class-organized before submitting an
ImageNet-C job. Keep ImageNet-C provenance and archives beside the shared
dataset, not in source or scratch outputs.

Generic one-off job (Isambard):

```bash
/lus/lfs1aip2/projects/u6gd/jiantao/SimLAP/cluster/submit.sh --job-name=ssl-smoke -- \
  bash /lus/lfs1aip2/projects/u6gd/jiantao/SimLAP/cluster/jobs/evaluate_ssl.sh
```

After placing datasets and checkpoints, submit a one-model smoke test
(one shot, seed 0):

```bash
/lus/lfs1aip2/projects/u6gd/jiantao/SimLAP/cluster/submit_ssl_comparison.sh smoke
```

Submit the architecture-matched transfer and ImageNet-C matrix with three
fixed seeds using:

```bash
/lus/lfs1aip2/projects/u6gd/jiantao/SimLAP/cluster/submit_ssl_comparison.sh final
```

Each batch job writes metrics and embeddings below the output root, then runs
`validate_results.py`. Validation requires schema version `1.0.0`, matching
manifest identity and checkpoint metadata, complete shot/seed coverage, and
the full 15-corruption/5-severity ImageNet-C protocol. Invalid records fail
the batch job and must not be passed to a later aggregator. The launchers do
not generate reports. Aggregate validated records after jobs finish:

```bash
python aggregate_report.py \
  --manifest model_manifest.json \
  --results-dir /scratch/u6gd/jw02425.u6gd/SimLAP/ssl-representation-comparison/outputs/metrics \
  --output-dir /scratch/u6gd/jw02425.u6gd/SimLAP/ssl-representation-comparison/outputs/report
```

This writes `summary.json`, `summary.csv`, and `report.md` under the supplied
output directory. ResNet and ViT groups remain separate. Missing or
non-default shot/seed evaluations are marked `incomplete`; their metrics stay
absent rather than being inferred. For a local smoke check, pass individual
JSON paths instead of `--results-dir` and write to a temporary or `results/`
directory so generated files stay out of the source tree.

## Hydra-MoCo for Arbitrary Contrastive Leraning with ResNet and ViT

### Introduction
This is a PyTorch implementation of [Hydra-MoCo](https://arxiv.org/abs/2410.18200) for Arbitrary Contrastive Leraning with ResNet and ViT.

This repository is based on the [MoCo v3](https://arxiv.org/abs/2104.02057) and [codes](https://github.com/facebookresearch/moco-v3).
### Main Results

The following results are based on ImageNet-1k self-supervised pre-training, followed by ImageNet-1k supervised training for linear evaluation or end-to-end fine-tuning. All results in these tables are based on a batch size of 4096.

**Pre-trained models** and **configs** can be found at [CONFIG.md](CONFIG.md). 

| ft.        | IN1K | note | ckpt |
|------------|------|------|------|
| mocov3     |      |      |      |
| hydra_moco | [84.02](https://wandb.ai/dlib/hydra/runs/7drtyzly/overview)|Trained 200 epochs from MAE-pretrained weights |      |

### Usage: Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo v1/2](https://github.com/facebookresearch/moco), this repo contains minimal modifications on the official PyTorch ImageNet code. We assume the user can successfully run the official PyTorch ImageNet code.
For ViT models, install [timm](https://github.com/rwightman/pytorch-image-models) (`timm==0.4.9`).

The code has been tested with CUDA 10.2/CuDNN 7.6.5, PyTorch 1.9.0 and timm 0.4.9.

### Usage: Self-supervised Pre-Training

Below are three examples for MoCo v3 pre-training. 

#### ResNet-50 with 2-node (16-GPU) training, batch 4096

On the first node, run:
```
torchrun --nproc_per_node=8 --nnodes=2  --node-rank=${rank} main_moco.py \
  --moco-m-cos --crop-min=.2 \
  [your imagenet-folder with train and val folders]
```
On the second node, run the same command with `--rank 1`.
With a batch size of 4096, the training can fit into 2 nodes with a total of 16 Volta 32G GPUs. 


#### ViT-Small with 1-node (8-GPU) training, batch 1024

```
torchrun --nproc_per_node=8 --nnodes=1  --node-rank=${rank} main_moco.py \
  -a vit_small -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  [your imagenet-folder with train and val folders]
```

#### ViT-Base with 8-node training, batch 4096

With a batch size of 4096, ViT-Base is trained with 8 nodes:
```
torchrun --nproc_per_node=8 --nnodes=2  --node-rank=${rank} \
  main_moco.py -a vit_base \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --gin MoCo.beta=1 MoCo.sep=True \
  [your imagenet-folder with train and val folders]
```
On other nodes, run the same command with `--rank 1`, ..., `--rank 7` respectively.


#### selective class pairs
We set a sampling weight for each pair $\frac{1}{(1+d)^{\alpha}}$, where $d$ denotes the distance of the pair in the semantic hierarchy (WordNet), and $\alpha$ is a hyperparameter to adjust the weight.  


Each class pair has the same sampling weight and will be equally sampled
```
torchrun --nproc_per_node=8 \
  main_moco.py -a vit_base -b 2048\
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --gin MoCo.beta=1 MoCo.sep=True PairSampler.alpha=0 \
  --output_dir outputs/positive/a0 \ 
  [your imagenet-folder with train and val folders]
```

The semantic distance between samples in a pair must be smaller than 1, that is, only identical class pairs are sampled:
```
torchrun --nproc_per_node=8 \
  main_moco.py -a vit_base -b 2048\
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --gin MoCo.beta=1 MoCo.sep=True PairSampler.max_level=1 PairSampler.alpha=0 \
  --output_dir outputs/positive/a10 \ 
  [your imagenet-folder with train and val folders]
```

The close pairs has a higher chance to be sampled:
```
torchrun --nproc_per_node=8 \
  main_moco.py -a vit_base -b 2048\
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --gin MoCo.beta=1 MoCo.sep=True PairSampler.max_level=11 PairSampler.alpha=2 \
  --output_dir outputs/positive/a2 \ 
  [your imagenet-folder with train and val folders]
```

#### Notes:
1. The batch size specified by `-b` is the total batch size across all GPUs.
1. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677) in [this line](https://github.com/facebookresearch/moco-v3/blob/main/main_moco.py#L213).
1. Using a smaller batch size has a more stable result (see paper), but has lower speed. Using a large batch size is critical for good speed in TPUs (as we did in the paper).
1. In this repo, only *multi-gpu*, *DistributedDataParallel* training is supported; single-gpu or DataParallel training is not supported. This code is improved to better suit the *multi-node* setting, and by default uses automatic *mixed-precision* for pre-training.

### Usage: Linear Classification

By default, we use momentum-SGD and a batch size of 1024 for linear classification on frozen features/weights. This can be done with a single 8-GPU node.

```
python main_lincls.py \
  -a [architecture] --lr [learning rate] \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/[your checkpoint file].pth.tar \
  [your imagenet-folder with train and val folders]
```

### Usage: End-to-End Fine-tuning ViT

To perform end-to-end fine-tuning for ViT, use our script to convert the pre-trained ViT checkpoint to [DEiT](https://github.com/facebookresearch/deit) format:
```
python convert_to_deit.py \
  --input [your checkpoint path]/[your checkpoint file].pth.tar \
  --output [target checkpoint file].pth
```
Then run the training (in the DeiT repo) with the converted checkpoint:
```
python $DEIT_DIR/main.py \
  --resume [target checkpoint file].pth \
  --epochs 150
```
This gives us 83.2% accuracy for ViT-Base with 150-epoch fine-tuning.

**Note**:
1. We use `--resume` rather than `--finetune` in the DeiT repo, as its `--finetune` option trains under eval mode. When loading the pre-trained model, revise `model_without_ddp.load_state_dict(checkpoint['model'])` with `strict=False`.
1. Our ViT-Small is with `heads=12` in the Transformer block, while by default in DeiT it is `heads=6`. Please modify the DeiT code accordingly when fine-tuning our ViT-Small model. 

### Model Configs

See the commands listed in [CONFIG.md](CONFIG.md) for specific model configs, including our recommended hyper-parameters and pre-trained reference models.

### Transfer Learning

See the instructions in the [transfer](https://github.com/facebookresearch/moco-v3/tree/main/transfer) dir.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation
```
@misc{wu2024rethinkingpositivepairscontrastive,
      title={Rethinking Positive Pairs in Contrastive Learning}, 
      author={Jiantao Wu and Shentong Mo and Zhenhua Feng and Sara Atito and Josef Kitler and Muhammad Awais},
      year={2024},
      eprint={2410.18200},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.18200}, 
}
```

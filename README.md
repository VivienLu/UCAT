# Calibrating Uncertainty for Zero-Shot Adversarial CLIP

[![ICML2026](https://img.shields.io/badge/ICML-2026-blue)](https://arxiv.org/abs/2512.12997)

Pytorch implementation of our method for ICML 2026 paper: "Calibrating Uncertainty for Zero-Shot Adversarial CLIP".

## Contents

- [Abstract](##Abstract)
- [Environment](##Environment)
- [Repository Structure](##RepositoryStructure)
- [Data Preparation](##DataPreparation)
- [Usage](##Usage)
- [Acknowledgment](##Acknowledgment)

## Abstract

![avatar](./image/framework.png)

CLIP delivers strong zero-shot classification but remains highly vulnerable to adversarial attacks. Prior adversarial fine-tuning work largely focuses on matching the predicted logits between clean and adversarial examples, which overlooks uncertainty calibration and may degrade the zero-shot generalization. A common expectation in reliable uncertainty estimation is that predictive uncertainty should increase as inputs become more difficult or shift away from the training distribution. However, we frequently observe the opposite in the adversarial setting: perturbations not only degrade accuracy but also suppress uncertainty, leading to severe miscalibration and unreliable over-confidence. This overlooked phenomenon highlights a critical reliability gap beyond robustness. To bridge this gap, we propose a novel adversarial fine-tuning objective for CLIP considering both prediction accuracy and uncertainty alignments. By reparameterizing the output of CLIP as the concentration parameter of a Dirichlet distribution, we propose a unified representation that captures relative semantic structure and confidence magnitude. Our objective aligns these distributions holistically under perturbations, moving beyond single-logit anchoring and restoring calibrated uncertainty. Experiments on multiple zero-shot classification benchmarks demonstrate that our approach effectively restores calibrated uncertainty and achieves competitive adversarial robustness while maintaining clean accuracy.

## Environment

Following the setup style used in TGA-ZSR, you can create a Python environment with `virtualenv`:

```bash
pip install virtualenv
virtualenv ucat
source ucat/bin/activate
pip install -r requirements.txt
```

If you prefer conda, an example environment file is also included:

```bash
conda env create -f environment.yml
conda activate zsar
pip install -r requirements.txt
```

Recommended software stack from the current repository:

- Python 3.8
- PyTorch 2.4.1
- TorchVision 0.19.1
- CUDA-enabled GPU

## Repository Structure

```text
UCAT/
├── main.py                  # single-label training and evaluation
├── main-multilabel.py       # multi-label evaluation
├── attacks.py               # attacks for single-label experiments
├── attacks_multilabel.py    # attacks for multi-label experiments
├── utils.py                 # dataset loading, prompts, logging, metrics
├── main.sh                  # example commands
├── models/                  # CLIP adaptation modules
├── replace/                 # customized CLIP and dataset wrappers
├── slip/                    # SLIP-related components
└── torchattacks/            # bundled attack implementations
```

## Data Preparation

The code supports the following datasets in the current implementation:

- Training: `tinyImageNet`, `ImageNet`
- Evaluation: `tinyImageNet`, `cifar10`, `cifar100`, `STL10`, `Food101`, `oxfordpet`, `flowers102`, `dtd`, `EuroSAT`, `fgvc_aircraft`, `Caltech101`, `Caltech256`, `StanfordCars`, `PCAM`, `ImageNet`, `SUN397`
- Multi-label evaluation: `coco2017`

Important note: dataset paths in [`utils.py`](/Users/luwenjing/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/ICML2026/UCAT/utils.py) are currently **hard-coded** for the original training environment. Before running experiments, please update those paths to your local dataset locations.

## Usage

### Training

Example single-label adversarial fine-tuning:

```bash
python -u ./main.py \
  --Method UCAT \
  --dataset tinyImageNet \
  --testdata tinyImageNet \
  --gpu 0 \
  --attack pgd \
  --train_eps 1 \
  --train_numsteps 2 \
  --train_stepsize 1 \
  --test_eps 1 \
  --test_numsteps 100 \
  --test_stepsize 1 \
  --target_concentration 0.07 \
  --batch_size 256 \
  --save_dir ./results/
```

### Evaluation

Single-label evaluation from a checkpoint:

```bash
python -u ./main.py \
  --mode test \
  --Method UCAT \
  --testdata dtd \
  --gpu 0 \
  --attack pgd \
  --test_eps 1 \
  --test_numsteps 100 \
  --batch_size 256 \
  --target_concentration 0.07 \
  --save_dir ./results/ \
  --resume /path/to/model_best.pth.tar
```

Multi-label evaluation on COCO 2017:

```bash
python -u ./main-multilabel.py \
  --mode test \
  --Method UCAT \
  --testdata coco2017 \
  --dataset tinyImageNet \
  --gpu 0 \
  --attack pgd \
  --test_eps 1 \
  --test_numsteps 100 \
  --batch_size 256 \
  --target_concentration 0.07 \
  --save_dir ./results/ \
  --resume /path/to/model_best.pth.tar
```

You can also use the provided script:

```bash
bash ./main.sh
```

### Main Arguments

- `--Method`: experiment name used in log and checkpoint folders
- `--dataset`: training dataset
- `--testdata`: evaluation dataset(s)
- `--attack`: attack type, such as `pgd`, `CW`, `autoattack`, `CAA`, or `a3`
- `--train_eps`: adversarial perturbation budget for training
- `--train_numsteps`: number of training attack iterations
- `--train_stepsize`: training attack step size
- `--test_eps`: adversarial perturbation budget for evaluation
- `--test_numsteps`: number of evaluation attack iterations
- `--test_stepsize`: evaluation attack step size
- `--target_concentration`: uncertainty calibration temperature/concentration parameter
- `--tau`: robustness loss weight scaling
- `--resume`: checkpoint path
- `--mode`: `train` or `test`
- `--save_dir`: directory for logs and checkpoints

### Notes

- `main.py` is the primary entry point for single-label experiments.
- `main-multilabel.py` is mainly used for COCO multi-label evaluation.
- The code keeps a frozen CLIP copy as the reference model during training and evaluation.
- Results, logs, and copied launch scripts are saved under `save_dir/Method/`.

## Acknowledgement

This repository builds upon prior open-source projects, especially:

- [TeCoA](https://github.com/cvlab-columbia/ZSRobust4FoundationModel)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [AutoAttack](https://github.com/fra31/auto-attack)
- [TGA-ZSR](https://github.com/zhyblue424/TGA-ZSR)
- [FARE](https://github.com/chs20/RobustVLM)

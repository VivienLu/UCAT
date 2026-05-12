# Calibrating Uncertainty for Zero-Shot Adversarial CLIP

This repository contains the official code for **Calibrating Uncertainty for Zero-Shot Adversarial CLIP (UCAT)**.

- Paper: [arXiv:2512.12997](https://arxiv.org/abs/2512.12997)
- Authors: Wenjing Lu, Zerui Tao, Dongping Zhang, Yuning Qiu, Yang Yang, Qibin Zhao
- Base code: this implementation is developed on top of **TGA-ZSR**

UCAT improves zero-shot adversarial robustness for CLIP by calibrating the uncertainty of adversarial predictions against a frozen clean reference model. The current codebase includes:

- Single-label training and evaluation in [`main.py`](/Users/luwenjing/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/ICML2026/UCAT/main.py)
- Multi-label evaluation in [`main-multilabel.py`](/Users/luwenjing/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/ICML2026/UCAT/main-multilabel.py)
- Adversarial attacks including `pgd`, `CW`, `autoattack`, `CAA`, and `a3`

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

- Training: `tinyImageNet`, `cifar10`, `cifar100`, `ImageNet`
- Evaluation: `tinyImageNet`, `cifar10`, `cifar100`, `STL10`, `Food101`, `oxfordpet`, `flowers102`, `dtd`, `EuroSAT`, `fgvc_aircraft`, `Caltech101`, `Caltech256`, `StanfordCars`, `PCAM`, `ImageNet`, `SUN397`
- Multi-label evaluation: `coco2017`

Important note: dataset paths in [`utils.py`](/Users/luwenjing/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/ICML2026/UCAT/utils.py) are currently **hard-coded** for the original training environment. Before running experiments, please update those paths to your local dataset locations.

## Training

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

## Evaluation

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

## Main Arguments

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

## Notes

- `main.py` is the primary entry point for single-label experiments.
- `main-multilabel.py` is mainly used for COCO multi-label evaluation.
- The code keeps a frozen CLIP copy as the reference model during training and evaluation.
- Results, logs, and copied launch scripts are saved under `save_dir/Method/`.

## Acknowledgement

This repository builds upon prior open-source projects, especially:

- [TeCoA](https://github.com/cvlab-columbia/ZSRobust4FoundationModel)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [AutoAttack](https://github.com/fra31/auto-attack)

## Citation

If you find this repository useful, please cite:

```bibtex
@article{ucat2025,
  title={Calibrating Uncertainty for Zero-Shot Adversarial CLIP},
  author={Lu, Wenjing and Tao, Zerui and Zhang, Dongping and Qiu, Yuning and Yang, Yang and Zhao, Qibin},
  journal={arXiv preprint arXiv:2512.12997},
  year={2025}
}
```

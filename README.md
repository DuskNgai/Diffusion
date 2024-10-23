# Diffusion

## Setup

First, download the codebase and the submodules:

```bash
git clone https://github.com/DuskNgai/diffusion.git -o diffusion && cd diffusion
git submodule update --init --recursive
```

Second, install the dependencies by manually installing them:

```bash
conda create -n diffusion python=3.11
conda activate diffusion
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ipykernel lightning matplotlib "numpy<2.0.0" pandas rich tensorboard
pip install deepspeed diffusers fvcore omegaconf timm
```

## Usage

We have a unified training command for all the experiments:

```bash
python train.py \
--config-file <PATH_TO_CONFIG_FILE> \
--num-gpus <NUM_GPUS> \
--num-nodes <NUM_NODES> \
<KEY_TO_MODIFY> <VALUE_TO_MODIFY>
```

We recommend naming the configuration file and output directory with the following format:
```txt
Configuration file: <MODEL_NAME>_<PREDICTION_TYPE>_<DATASET_NAME>.yaml
Output directory: output/<MODEL_NAME>_<PREDICTION_TYPE>_<DATASET_NAME>
```

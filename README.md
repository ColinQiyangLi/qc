<div align="center">

# [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969)

## [[website](https://colinqiyangli.github.io/qc/)]      [[pdf](https://arxiv.org/pdf/2507.07969)]

</div>

<p align="center">
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="teaser figure" src="./assets/teaser.png" width="48%">
  </a>
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="aggregated results" src="./assets/agg.png" width="48%">
  </a>
</p>


## Overview
Q-chunking runs RL on a *temporally extended action (action chunking) space* with an expressive behavior constraint to leverage prior data for improved exploration and online sample efficiency.

## Installation
```bash
pip install -r requirements.txt
```


## Datasets
For robomimic, we assume the datasets are located at `~/.robomimic/lift/mh/low_dim_v15.hdf5`, `~/.robomimic/can/mh/low_dim_v15.hdf5`, and `~/.robomimic/square/mh/low_dim_v15.hdf5`. The datasets can be downloaded from https://robomimic.github.io/docs/datasets/robomimic_v0.1.html (under Method 2: Using Direct Download Links - Multi-Human (MH)).

For cube-quadruple, we use the 100M-size offline dataset. It can be downloaded from https://github.com/seohongpark/horizon-reduction via
```bash
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0/
```
and include this flag in the command line `--ogbench_dataset_dir=[realpath/to/your/cube-quadruple-play-100m-v0/]` to make sure it is using the 100M-size dataset.

## Reproducing paper results

We include the example command for all the methods we evaluate in our paper below. For `scene` and `puzzle-3x3` domains, use `--sparse=True`. We also release our plot data at [plot_data/README.md](plot_data/README.md).

For Mac you have to set

```bash
MUJOCO_GL=glfw
```

For validating with ogbench dataset:

```bash
MUJOCO_GL=glfw uv run python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --ogbench_dataset_dir=[realpath/to/your/cube-quadruple-play-100m-v0/]
```

```bash
# QC
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# BFN-n
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=4 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# BFN
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=4 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1

# QC-FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# FQL-n
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1

# RLPD
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1 

# RLPD-AC
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# QC-RLPD
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.bc_alpha=0.01
```

```
@inproceedings{
li2025reinforcement,
title={Reinforcement Learning with Action Chunking},
author={Qiyang Li and Zhiyuan Zhou and Sergey Levine},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=XUks1Y96NR}
}
```

## Acknowledgments
This codebase is built on top of [FQL](https://github.com/seohongpark/fql). The two rlpd_* folders are directly taken from [RLPD](https://github.com/ikostrikov/rlpd).

# Isaac Gym RL template environment for legged robots
This repository contains an Isaac Gym template environment that can be used to train any legged robot using [rl_games](https://github.com/Denys88/rl_games). This repository is deployed with zero-shot sim-to-real transfer in the following projects:
- [Text2Robot](https://github.com/generalroboticslab/Text2Robot): a framework that converts user text specifications and performance preferences into physical quadrupedal robots.

  <img src="https://github.com/generalroboticslab/Text2Robot/blob/master/figures/teaser.gif" alt="teaser" style="width:20%; margin: 0;">
- [DukeHumanoidv1](https://github.com/generalroboticslab/DukeHumanoidv1): an open-source 10-degrees-of-freedom child size humanoid for locomotion research.

  <img src="https://github.com/generalroboticslab/dukeHumanoidHardwareControl/blob/master/doc/image/dukehumanoidv1-thumbnails_1.gif" alt="teaser" style="width:20%; margin: 0;">


<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->
## Table of content
- [project structure](#project-structure)
- [Setting up](#setting-up)
   * [Tested environment](#tested-environment)
   * [Setup python virtual environment](#setup-python-virtual-environment)
   * [Setup vscode](#setup-vscode)
   * [to start conda env](#to-start-conda-env)
   * [to train a policy](#to-train-a-policy)
   * [to run example checkpoint:](#to-run-example-checkpoint)

<!-- TOC end -->

## project structure
```
.
├── assets
│   ├── checkpoints # saved checkpoint
│   │   ├── A1Terrain.pth
│   │   └── ...
│   ├── joint_monkey.py # script to check the urdf
│   └── urdf
│       ├── a1
│       ├── anymal_c
│       └──...
├── envs
│   ├── cfg # stores all configurations
│   │   ├── config.yaml
│   │   ├── pbt
│   │   ├── task
│   │   └── train
│   ├── common # common scripts
│   │   ├── publisher.py
│   │   ├── terrain.py
│   │   └── utils.py
│   ├── exp.sh # experiment scripts
│   ├── __init__.py
│   ├── plot_juggler # plot_juggler configurations
│   │   └── robotdog7kg_play_debug.xml
│   ├── run.sh # run script
│   ├── setup # conda env setup script
│   │   ├── conda_env.yaml
│   │   └── ...
│   ├── tasks # 
│   │   ├── __init__.py
│   │   └── legged_terrain.py
│   └── train.py
├── outputs # contains trained results
└── README.md
```

## Setting up

### Tested environment
- Operating system: Ubuntu 22.04 with CUDA 12.3
- Development environment: Visual Studio Code (VSCode)
- Python environment management: micromanba

### Setup python virtual environment
 first [install Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) if you have not done so. Recommended to install at `~/repo/micromamba`


```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Setup a python virtual envirnpoment named  `py38` with conda yaml file `setup/conda_env.yaml` 

```bash
alias conda="micromamba"

# Create environment
conda env create --file setup/conda_env_py38.yaml -y

# Activate the environment
conda activate py38

# Export library path
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
```
Install addional dependencies:
```bash
conda activate py38 && export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

# install Isaacgym
cd ~/repo
git clone https://github.com/boxiXia/isaacgym.git
cd ~/repo/isaacgym/python
python -m pip install -e .

## verify
cd ~/repo/isaacgym/python/examples
python interop_torch.py
# if error: crypt.h: No such file or directory, do this:
# cp /usr/include/crypt.h /home/grl/repo/micromamba/envs/py38/include/python3.8/

# install rl_games
cd ~/repo
git clone https://github.com/Denys88/rl_games.git
cd ~/repo/rl_games/
# Optionally modify setup.py to remove the version control parts
python -m pip install -e .

# install IsaacGymEnvs
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/tree/main
cd ~/repo
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd ~/repo/IsaacGymEnvs
# recommended to modify setup.py to remove the version control parts
python -m pip install -e .

```

### Setup vscode
Install VSCode: [Download and install](https://code.visualstudio.com/download) vscode if you have not done so

Install vscode Extensions:
- Python: ms-python.python
Optionally Install other extensions such as Git based on your needs.

To configure the project using vscode:
- Open the project folder.
- Select the correct Python virtual environment.


### to start conda env
Ensure you're operating within the Python virtual environment.  For convenience, **we'll assume all commands in this guide are executed within this activated environment**:
```
conda activate py38 && export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
```

### to train a policy

```bash
cd envs

# duke humanoid v1 baseline
bash run.sh dukehumanoid_baseline

# duke humanoid v1 passive policy
bash run.sh dukehumanoid_passive

# robot dog7kg
bash run.sh dog

# a1
bash run.sh a1Terrain

# anymal
bash run.sh anymalTerrain
```


### to run example checkpoint:

```bash
cd envs

# duke humanoid v1 baseline
bash run.sh dukehumanoid_baseline -p
bash run.sh dukehumanoid_baseline -pk # with keyboard wasd

# duke humanoid v1 passive policy
bash run.sh dukehumanoid_passive -p

# robot dog7kg
bash run.sh dog -p
# a1
bash run.sh a1Terrain -p
# anymal
bash run.sh anymalTerrain -p

# dry run (print the command only)
bash run.sh anymalTerrain -d
```

### Debugging

Use `bash run.sh [THE_TASK_TO_RUN] -d` to start a remote attache python debuging using debugpy with vscode, for example

```bash 
bash run.sh dukehumanoid_baseline -d # train mode + debug
bash run.sh dukehumanoid_baseline -pd # play mode + debug
bash run.sh a1Terrain -pkd # play mode + keyboard + debug
```
Then go to the vscode debug panel (keyboard shortcut: `ctrl+shift+D`), in the `RUN AND DEBUG`, select option `Python Debugger: Remote Attach` to start debugging.

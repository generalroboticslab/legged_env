

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

### Setup vscode
Install VSCode: [Download and install](https://code.visualstudio.com/download) vscode if you have not done so

Install vscode Extensions:
- Python: ms-python.python
Optionally Install other extensions such as Git based on your needs.

To configure the project using vscode:
- Open the project folder.
- Select the correct Python virtual environment.


### to start conda env
first make sure you are in the python virtural environment
```
conda activate py38 && export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
```

### to train a policy:

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

```




# project structure
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
│   │   ├── terrrain.py
│   │   └── utils.py
│   ├── exp.sh # experiment scripts
│   ├── __init__.py
│   ├── plot_juggler # plot_juggler configurations
│   │   └── robotdog7kg_play_debug.xml
│   ├── run.sh # run script
│   ├── setup # conda env setup script
│   │   ├── conda_env.yaml
│   │   └── plotjuggler_pipeline_debug.xml
│   ├── tasks # 
│   │   ├── __init__.py
│   │   └── quadruped_terrain.py
│   └── train.py
├── outputs # contains trained results
└── README.md
```

# to run example:

```bash
cd envs
# robot dog7kg
bash run.sh dog -p
# a1
bash run.sh a1Terrain -p
# anymal
bash run.sh anymalTerrain -p
```
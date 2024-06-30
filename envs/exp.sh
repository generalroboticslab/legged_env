# #!/bin/bash

biped_c2r16(){ # bad gait, same as biped_c2r15 but changed level up to be 90% complete of commanded value
    biped_c2r15 
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r16/runs/BipedAsymm_30-01-25-01/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_c2r16/runs/BipedAsymm_30-01-25-01/nn/last_BipedAsymm_ep_3550_rew_26.177423.pth
    
    )
}

biped_c2r15(){ # bad gait, terrain curriculum 0.8
    biped_c2r13
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r15/runs/BipedAsymm_30-01-20-53/nn/BipedAsymm.pth
    )
    TRAIN_ARGS+=(
        train.params.config.max_epochs=5000
        checkpoint=null
    )
    BASE_ARGS+=(
    task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.5]
    task.env.randomCommandVelocityRanges.linear_y=[0,0]
    task.env.randomCommandVelocityRanges.yaw=[0,0] # no yaw
    )
}

biped_c2r14(){ # bad gait, terrain curriculum 0.8, max_epoch=10000
    biped_c2r13
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r14/runs/BipedAsymm_30-01-16-46/nn/BipedAsymm.pth
    )
    TRAIN_ARGS+=(
        train.params.config.max_epochs=10000
        checkpoint=null
    )
}

biped_c2r13(){ # straight leg gait, warm start, terrain curriculum 0.8
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r13/runs/BipedAsymm_30-01-14-36/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=false
        num_envs=20
    )
    TRAIN_ARGS+=(
        # train.params.config.max_epochs=10000
        checkpoint=outputs/Biped/train/biped_c2r10/runs/BipedAsymm_30-00-50-11/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation removes linearVelocity, states add contact (same as biped_c2r10)
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contact]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1

        task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[-3.142,3.142] # biped_c2r12 add rotation

        task.env.terrain.difficultySale=0.2
        task.env.terrain.terrainType=heightfield
         # terrain types: [rough_up✔️, rough_down✔️, rough_flat✔️, stair_up, stair_down, smooth_up✔️, smooth_down✔️, discrete, stepping_stone]
        task.env.terrain.terrainProportions=[1,1,1,0,0,0,1,1,0,0]

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


biped_c2r12(){ # small steps, observation removes linearVelocity, states add contact (same as biped_c2r10)
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r12/runs/BipedAsymm_30-01-08-49/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contact]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1

        task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[-3.142,3.142] # biped_c2r12 add rotation

        # train.params.config.max_epochs=5000

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


biped_c2r11(){ # add contact to observation result in very small steps! 
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r11/runs/BipedAsymm_30-00-57-46/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation removes linearVelocity, states and observation add contact
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contact]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contact]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


biped_c2r10(){ # add contact to states helps
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r10/runs/BipedAsymm_30-00-50-11/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation does not have linearVelocity, states add contact
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contact]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}

biped_c2r9(){ # baseHeight in state result in very small steps
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r9/runs/BipedAsymm_30-00-43-57/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation does not have linearVelocity, states add baseHeight
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,baseHeight]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}



biped_c2r8(){
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        # bad gait, worse than biped_c2r5
        checkpoint=outputs/Biped/train/biped_c2r8/runs/BipedAsymm_29-20-27-06/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env

    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation does not have linearVelocity
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,dofForce,dofForceTarget,actions]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


biped_c2r7(){
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        # small steps, worse than biped_c2r5
        checkpoint=outputs/Biped/train/biped_c2r7/runs/BipedAsymm_29-20-26-55/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env

    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation does not have linearVelocity
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,dofForceTarget,actions]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


biped_c2r6(){
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        # similar to biped_c2r5
        checkpoint=outputs/Biped/train/biped_c2r6/runs/BipedAsymm_29-20-26-48/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env

    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation does not have linearVelocity
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,dofForce,actions]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1
        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


biped_c2r5(){ # 👍use this as baseline, asymmetric obs, feetSlip=-0.1
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(

        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
        
        checkpoint=outputs/Biped/train/biped_c2r5/runs/BipedAsymm_29-18-21-24/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation does not have linearVelocity
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]

        # same as biped_c2r3
        task.env.learn.reward.feetSlip.scale=-0.1

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}

biped_c2r4(){
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(

        # checkpoint=outputs/Biped/train/biped_c2r4/runs/BipedAsymm_29-17-59-32/nn/BipedAsymm.pth # mixed_precision: True
        checkpoint=outputs/Biped/train/biped_c2r4/runs/BipedAsymm_29-17-59-56/nn/BipedAsymm.pth # mixed_precision: False  slightly better
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True

        task.env.learn.reward.feetSlip.scale=-0.1
    )
}



biped_c2r3(){
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r3/runs/Biped_29-17-14-05/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.feetSlip.scale=-0.1
    )
}

biped_c2r2(){
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(
        # ok
        checkpoint=outputs/Biped/train/biped_c2r2_20240629_163756/runs/Biped_29-16-37-56/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.feetSlip.scale=-0.1
    )
}

biped_c2r1(){
    biped_c2
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/20240629_161503/runs/Biped_29-16-15-03/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.feetSlip.scale=-0.05
        # change_hydra_dir
        "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


biped_c2(){
    biped

    PLAY_ARGS+=( # control variation
        # bad orientation, robot leaning sideways
        checkpoint=outputs/Biped/train/20240629_161016/runs/Biped_29-16-10-16/nn/Biped.pth
    )
    TRAIN_ARGS+=(
        headless=true
    )

    BASE_ARGS+=(
        task.env.control.stiffness=60.0
        task.env.control.actionScale=1.0
    )
}


# biped_c1(){ # command/control variation
#     biped
#     PLAY_ARGS+=(
#         checkpoint=outputs/Biped/train/20240629_132037/runs/Biped_29-13-20-37/nn/Biped.pth
#     )
#     TRAIN_ARGS+=(
#         checkpoint=outputs/Biped/train/20240629_023908/runs/Biped_29-02-39-08/nn/Biped.pth
#     )
#     BASE_ARGS+=(
#         task.env.randomCommandVelocityRanges.linear_x=[0,0.5]
#         task.env.randomCommandVelocityRanges.linear_y=[0,0]
#         task.env.randomCommandVelocityRanges.yaw=[0,0]
#     )
# }


biped_o1(){
    biped
    PLAY_ARGS+=( # observation variation experiment
        checkpoint=outputs/Biped/train/20240629_022324/runs/Biped_29-02-23-25/nn/Biped.pth
    )
    BASE_ARGS+=(
        "task.env.observationNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,dofForce,actions]"
    )
}

biped_v(){
    biped
    BASE_ARGS+=(
    task.env.urdfAsset.file=urdf/biped_visual/biped_v6.urdf
    )
}

biped_a2(){
    biped

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_a1/runs/Biped_29-18-58-45/nn/Biped.pth
    )
    BASE_ARGS+=(
        ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0]
        ++task.env.initialJointPositions=[0,0,0,0,0,0,0,0,0,0]

        "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


biped_a1(){
    biped

    PLAY_ARGS+=(
        # bad gait, this means defaultJointPositions should be chosen carefully
        checkpoint=outputs/Biped/train/biped_a1/runs/Biped_29-18-58-45/nn/Biped.pth
    )
    BASE_ARGS+=(
        ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0]
        ++task.env.initialJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]

        "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}

biped_hang(){
    biped
    BASE_ARGS+=(
        ++task.env.baseHeightOffset=1
        task.env.urdfAsset.AssetOptions.fix_base_link=true
    )
}

biped(){
    # remove heightmap
    base
    task=Biped
    TRAIN_ARGS+=(
        train.params.config.max_epochs=5000
        headless=true
    )
    PLAY_ARGS+=(

        num_envs=2

        # test=export
        # num_envs=1 # exported policy only works with 1 num of env

        checkpoint=outputs/Biped/train/biped/runs/Biped_29-18-50-06/nn/Biped.pth # new enlarged cmd x
        
        task.env.dataPublisher.enable=true

        task.sim.dt=0.0025
        ++task.env.renderFPS=100

        # ++task.env.renderFPS=50

        task.env.learn.episodeLength_s=999
        # task.env.terrain.terrainType=plane
    )
    BASE_ARGS+=(
        task.sim.physx.num_position_iterations=2 # reduced for faster training # works!
        # # dynamics matching
        # ++task.env.assetDofProperties.friction=0.008
        # ++task.env.assetDofProperties.armature=0.2

        ++task.env.assetDofProperties.damping=0.88
        ++task.env.assetDofProperties.friction=0.00345
        ++task.env.assetDofProperties.armature=0.25
        ++task.env.assetDofProperties.velocity=10

        ++task.env.baseHeightOffset=0.05
        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]
        
        
        # task.env.randomCommandVelocityRanges.linear_x=[0.5,0.5]
        task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]

        task.env.terrain.terrainType=plane
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.difficultySale=0.2
        # task.env.terrain.curriculum=true
        # task.env.terrain.terrainProportions=[1,1,1,0,0,0,1,1,0,0]

        "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
    )
}


shoe(){
    dog
    task=RobotDog
    PLAY_ARGS+=(
        checkpoint=outputs/RobotDog/False/20240613_024701/runs/RobotDog_13-02-47-01/nn/RobotDog.pth
    )
    
    BASE_ARGS+=(
    # task.env.terrain.terrainType=plane
    ++task.env.urdfAsset.root="../evolutionary_loop/assets"
    task.env.urdfAsset.file="ShoeBot_BugBot3_top-z-0_BreadBot_bottom-x-3/ShoeBot_BugBot3_top-z-0_BreadBot_bottom-x-3.urdf"
    )
}


frog(){
    dog
    task=RobotDog
    PLAY_ARGS+=(
        num_envs=1
        checkpoint=outputs/RobotDog/False/20240604_155930/runs/RobotDog_04-15-59-30/nn/RobotDog.pth
        # task.env.learn.episodeLength_s=5
    )
    BASE_ARGS+=(
    # task.env.terrain.terrainType=plane
    ++task.env.urdfAsset.root="../evolutionary_loop/assets"
    task.env.urdfAsset.file="FrogBot5_FrogBot1_top-z-0_FrogBot5_bottom-x-5/FrogBot5_FrogBot1_top-z-0_FrogBot5_bottom-x-5.urdf"
    )
}


cricket(){
    dog
    task=RobotDog
    PLAY_ARGS+=(
        num_envs=1
        checkpoint=outputs/RobotDog/False/20240603_173047/runs/RobotDog_03-17-30-47/nn/RobotDog.pth
        # task.env.learn.episodeLength_s=5
    )
    TRAIN_ARGS+=(
        # headless=false
    )
    BASE_ARGS+=(
    # task.env.terrain.terrainType=plane
    ++task.env.urdfAsset.root="../evolutionary_loop/assets"
    task.env.urdfAsset.file="CricketBot_RobotDog_top-z-1_CanBot_bottom-x-0/CricketBot_RobotDog_top-z-1_CanBot_bottom-x-0.urdf"
    )
}



robogram(){
    base
    task=RobotDog
    PLAY_ARGS+=(
        
        checkpoint=outputs/RobotDog/False/2024-05-23_10-20-38/runs/RobotDog_23-10-20-39/nn/RobotDog.pth
        num_envs=2
        task.env.dataPublisher.enable=true
        task.env.learn.episodeLength_s=5
    )

    BASE_ARGS+=(
    task.env.urdfAsset.collision_filter=1
    # task.env.urdfAsset.AssetOptions.collapse_fixed_joints=true
    task.env.urdfAsset.AssetOptions.collapse_fixed_joints=false
    ++task.env.urdfAsset.root=../evolutionary_loop/assets/robogrammar_bank
    task.env.urdfAsset.file="robot_1350/robot.urdf"
    task.env.terrain.terrainType=plane
    task.env.randomCommandVelocityRanges.linear_x=[-0.5,0.5]
    task.env.randomCommandVelocityRanges.linear_y=[-0.5,0.5]
    task.env.randomCommandVelocityRanges.yaw=[-1,1]
    )
}



dog(){
    base
    task=RobotDog
    PLAY_ARGS+=(
        checkpoint=assets/checkpoints/RobotDog.pth
        task.env.terrain.numLevels=3
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=false
        num_envs=20
        task.env.dataPublisher.enable=true
    )
    TRAIN_ARGS+=(
        # headless=false
    )
    BASE_ARGS+=(
        ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_convex_hulls=3
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_num_vertices_per_ch=16
        # task.env.terrain.terrainType=plane
        task.env.terrain.terrainType=heightfield
    )
}

a1(){
    # bash run.sh a1Terrain -p
    base
    task=A1Terrain
    PLAY_ARGS+=(
        num_envs=2
        checkpoint=assets/checkpoints/A1Terrain.pth
        task.env.dataPublisher.enable=true
    )
    BASE_ARGS+=(
        task.env.terrain.terrainType=plane
    )
}

a1Terrain(){
    # bash run.sh a1Terrain -p
    base
    task=A1Terrain
    PLAY_ARGS+=(
        checkpoint=assets/checkpoints/A1Terrain.pth
        # task.env.dataPublisher.enable=true
        # ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        task.env.enableDebugVis=True
        task.env.dataPublisher.enable=true
        num_envs=20
        task.env.terrain.terrainType=heightfield
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
    )
}


anymalTerrain(){
    # bash run.sh anymalTerrain -p
    base
    task=AnymalTerrain
    PLAY_ARGS+=(
        num_envs=2
        checkpoint=assets/checkpoints/AnymalTerrain.pth
        task.env.terrain.terrainType=plane
        # task.task.randomize=true
        # env.enableDebugVis=True
    )
}

base(){
    checkpoint=null
    ENTRY_POINT=train.py
    PLAY_ARGS=(
        test=play
    )
    
    TRAIN_ARGS=(
        headless=true
    )
    BASE_ARGS=()
    KEYBOARD_ARGS=(
        task.env.viewer.keyboardOperator=true
    )
}

change_hydra_dir(){
    BASE_ARGS+=(
        "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[1]}"
    )
}

match_dyn(){
    match
    BASE_ARGS+=(
        task.env.control.decimation=2
        ++task.env.renderFPS=100
        num_envs=1

        ++task.env.assetDofProperties.velocity=20
        ++task.env.assetDofProperties.stiffness=0

        # ++task.env.assetDofProperties.damping=0.275
        # ++task.env.assetDofProperties.friction=0.0055
        # ++task.env.assetDofProperties.armature=0.2

        ++task.env.assetDofProperties.damping=0.88
        ++task.env.assetDofProperties.friction=0.00345
        ++task.env.assetDofProperties.armature=0.25

        task.env.control.stiffness=80
        task.env.control.damping=2
        ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0]

        # ++task.env.defaultJointPositions=[0,1.047,0,0,0,0,-1.047,0,0,0]
        ++task.env.initialJointPositions=[0,0,0,0,0,0,0,0,0,0]
    )
}


match(){
    ENTRY_POINT=dyanmics_matching.py
    task=Biped

    BASE_ARGS+=(
        test=play

        num_envs=2
        headless=false
        task.env.dataPublisher.enable=true

        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.learn.addNoise=false

        task.env.terrain.terrainType=plane
        ++task.env.baseHeightOffset=1
        task.env.urdfAsset.AssetOptions.fix_base_link=true
        ++task.env.defaultJointPositions=[0,1.5708,0,0,0,0.000,-1.5708,0,0,0]

        task.env.control.stiffness=0
        task.env.control.damping=0

        pipeline="cpu"

        ++task.env.assetDofProperties.velocity=20
        ++task.env.assetDofProperties.stiffness=0
        ++task.env.assetDofProperties.damping=0
        ++task.env.assetDofProperties.friction=0.008
        ++task.env.assetDofProperties.armature=0.2

        ++task.env.renderFPS=50
        task.env.learn.episodeLength_s=555
    )
}


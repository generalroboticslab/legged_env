# #!/bin/bash






# t1(){
#     base
#     task=RobotDog
#     ENTRY_POINT="parallel_train.py"
#     BASE_ARGS+=(
#         task.env.urdfAsset.file="/home/grl/repo/legged/assets/urdf/pool_of_dog"
#     )
# }


# ant_pbt(){
#     base
#     task=RobotDog
#     ENTRY_POINT="-m isaacgymenvs.train"
#     BASE_ARGS+=(
#     )
# }

robogram_u1(){
    robogram
    PLAY_ARGS+=(
        checkpoint=null
    )

    BASE_ARGS+=(
        task.env.urdfAsset.file="robot_3104/robot.urdf"
    )

}

robogram(){
    base
    task=RobotDog
    PLAY_ARGS+=(
        
        # checkpoint=../outputs/RobotDog/False/2024-05-23_09-50-54/runs/RobotDog_23-09-50-54/nn/RobotDog.pth
        checkpoint=../outputs/RobotDog/False/2024-05-23_10-20-38/runs/RobotDog_23-10-20-39/nn/RobotDog.pth
        num_envs=2
        task.env.dataPublisher.enable=true
        pipeline="cpu"
        task.env.learn.episodeLength_s=5
        # task.env.randomCommandVelocityRanges.linear_x=[0.5,0.5]
        # task.env.randomCommandVelocityRanges.linear_y=[0.,0.]
        # task.env.randomCommandVelocityRanges.yaw=[1,1]
    )

    BASE_ARGS+=(
    task.env.urdfAsset.collision_filter=1
    task.env.urdfAsset.AssetOptions.collapse_fixed_joints=false
    # relative to the quadruped_terrain.py file
    ++task.env.urdfAsset.root=../evolutionary_loop/assets/robogrammar_bank
    task.env.urdfAsset.file="robot_1350/robot.urdf"
    task.env.terrain.terrainType=plane
    task.env.randomCommandVelocityRanges.linear_x=[-0.5,0.5]
    task.env.randomCommandVelocityRanges.linear_y=[-0.5,0.5]
    task.env.randomCommandVelocityRanges.yaw=[-1,1]
    )
}

dog3kgv1(){
    base
    task=RobotDog
    PLAY_ARGS+=(
        # checkpoint=../outputs/RobotDog/False/2024-05-17_11-13-14-3kg_x0.4/runs/RobotDog_17-11-13-15/nn/RobotDog.pth
        # checkpoint=../outputs/RobotDog/False/2024-05-17_11-31-35/runs/RobotDog_17-11-31-35/nn/RobotDog.pth
        checkpoint=../outputs/RobotDog/False/2024-05-17_11-36-03/runs/RobotDog_17-11-36-03/nn/RobotDog.pth
        checkpoint=../outputs/RobotDog/False/2024-05-17_11-39-56/runs/RobotDog_17-11-39-57/nn/RobotDog.pth
        num_envs=1
        task.env.dataPublisher.enable=true
        pipeline="cpu"
        task.env.learn.episodeLength_s=5
        # task.env.randomCommandVelocityRanges.linear_x=[0.5,0.5]
        # task.env.randomCommandVelocityRanges.linear_y=[0.,0.]
        # task.env.randomCommandVelocityRanges.yaw=[1,1]
    )
    BASE_ARGS+=(
    task.env.urdfAsset.file="urdf/RobotDog/RobotDog3kg.urdf"
    task.env.terrain.terrainType=plane
    task.env.randomCommandVelocityRanges.linear_x=[-0.5,0.5]
    task.env.randomCommandVelocityRanges.linear_y=[-0.5,0.5]
    task.env.randomCommandVelocityRanges.yaw=[-1,1]
    )
}

dog_m1(){ # variant
    base
    task=RobotDog

    PLAY_ARGS+=(
        checkpoint=/home/grl/repo/legged/outputs/RobotDog/False/2024-04-30_16-51-24/runs/RobotDog_30-16-51-24/nn/RobotDog.pth
        num_envs=2
        # task.env.terrain.terrainType=plane
        task.env.randomCommandVelocityRanges.linear_x=[0.2,0.2]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]
        task.env.dataPublisher.enable=True
        
        # # headless (server)
        # headless=true
        # graphics_device_id=-1
    )
    TRAIN_ARGS+=(
        # headless=false
        task.env.randomCommandVelocityRanges.linear_x=[0.2,0.2]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]
    )

    BASE_ARGS+=(
        "task.env.urdfAsset.file="original_dog.urdf""
        "++task.env.urdfAsset.root="/home/grl/repo/legged/assets/urdf/URDFsForBoxi/original_dog""
        "task.env.baseHeightTarget=0.22946011562866567"
        "task.env.baseInitState.pos=[0,0,0.22946011562866567]"
        "task.env.baseInitState.rot=[0,0,0,1]"
        "~task.env.defaultJointAngles"
        "~task.env.desiredJointAngles"
        "++task.env.defaultJointAngles.joint_1_0=0"
        "++task.env.defaultJointAngles.joint_1_1=0"
        "++task.env.defaultJointAngles.joint_1_2=0"
        "++task.env.defaultJointAngles.joint_1_3=0"
        "++task.env.defaultJointAngles.joint_2_0=0"
        "++task.env.defaultJointAngles.joint_2_1=0"
        "++task.env.defaultJointAngles.joint_2_2=0"
        "++task.env.defaultJointAngles.joint_2_3=0"
        "++task.env.desiredJointAngles.joint_1_0=0"
        "++task.env.desiredJointAngles.joint_1_1=0"
        "++task.env.desiredJointAngles.joint_1_2=0"
        "++task.env.desiredJointAngles.joint_1_3=0"
        "++task.env.desiredJointAngles.joint_2_0=0"
        "++task.env.desiredJointAngles.joint_2_1=0"
        "++task.env.desiredJointAngles.joint_2_2=0"
        "++task.env.desiredJointAngles.joint_2_3=0"
        # additionals
        "task.env.urdfAsset.hipName=body_1"
        "task.env.urdfAsset.kneeName=body_1"
        "task.env.urdfAsset.footName=body_2"
        "task.env.urdfAsset.hipJointName=joint_1"
        task.env.terrain.terrainType=plane
    )
}

dog3kg_t2(){
    dog3kg
    task=RobotDog
    # foot dragging, moves
    PLAY_ARGS+=(
        checkpoint=../outputs/RobotDog/False/2024-05-14_22-54-23-dog3kg_t2/runs/RobotDog_14-22-54-23/nn/RobotDog.pth
        num_envs=1

    )
    BASE_ARGS+=(
       task.sim.dt=0.00625
       task.env.control.decimation=4
    )
}


dog3kg_t1(){
    dog3kg
    task=RobotDog
    PLAY_ARGS+=(
        # does not move well
        checkpoint=../outputs/RobotDog/False/2024-05-14_22-42-39/runs/RobotDog_14-22-42-39/nn/RobotDog.pth
        num_envs=1
    )
    BASE_ARGS+=(
       task.sim.dt=0.01
       task.env.control.decimation=2
    )
}


dog3kg_r2(){
    dog3kg
    task=RobotDog
    PLAY_ARGS+=(
        # moves similarly compared to baseline
        checkpoint=../outputs/RobotDog/False/2024-05-14_16-54-47-dog3kg_r2/runs/RobotDog_14-16-54-47/nn/RobotDog.pth
        num_envs=1
    )
    BASE_ARGS+=(
        task.env.learn.reward.feetAirTime.scale=0.5
        task.env.learn.reward.feetStanceTime.scale=0.5
    )
}

dog3kg_r1(){
    dog3kg
    task=RobotDog
    PLAY_ARGS+=(
        # cannot move well, proves that air/stance time is needed, entropy increased
        checkpoint=../outputs/RobotDog/False/2024-05-14_16-48-54-dog3kg_r1/runs/RobotDog_14-16-48-54/nn/RobotDog.pth
        num_envs=1
    )
    BASE_ARGS+=(
        task.env.learn.reward.feetAirTime.scale=0
        task.env.learn.reward.feetStanceTime.scale=0
    )
}

dog3kg_p1(){
    dog3kg
    task=RobotDog
    PLAY_ARGS+=(
        checkpoint=../outputs/RobotDog/False/2024-05-14_16-20-09-dog3kg_p1/runs/RobotDog_14-16-20-09/nn/RobotDog.pth
        num_envs=1
    )
    BASE_ARGS+=(
    task.env.randomCommandVelocityRanges.linear_x=[0.3,0.3]
    task.env.randomCommandVelocityRanges.linear_y=[0,0]
    task.env.randomCommandVelocityRanges.yaw=[0,0]
    )
}

dog3kg(){
    base
    task=RobotDog
    PLAY_ARGS+=(
        # checkpoint=../assets/checkpoints/RobotDog.pth
        # checkpoint=../outputs/RobotDog/False/2024-05-14_15-20-17-dog3kg/runs/RobotDog_14-15-20-17/nn/RobotDog.pth
        # checkpoint=../outputs/RobotDog/False/2024-05-14_23-33-01-dog3kg/runs/RobotDog_14-23-33-01/nn/RobotDog.pth
        # checkpoint=../outputs/RobotDog/False/2024-05-15_17-58-21-dog3kg/runs/RobotDog_15-17-58-21/nn/RobotDog.pth
        checkpoint=../outputs/RobotDog/False/2024-05-15_18-19-59/runs/RobotDog_15-18-19-59/nn/RobotDog.pth
        num_envs=1
        task.env.dataPublisher.enable=true
        pipeline="cpu"
    )
    BASE_ARGS+=(
    task.env.urdfAsset.file="urdf/RobotDog/RobotDog7kg.urdf"
    task.env.terrain.terrainType=plane
    task.env.randomCommandVelocityRanges.linear_x=[0.2,0.2]
    task.env.randomCommandVelocityRanges.linear_y=[0,0]
    task.env.randomCommandVelocityRanges.yaw=[0,0]
    )
}


dog(){
    base
    task=RobotDog
    PLAY_ARGS+=(
        checkpoint=../assets/checkpoints/RobotDog.pth
        num_envs=1
        # task.env.terrain.terrainType=plane
        # task.env.dataPublisher.enable=True
        # # headless (server)
        # headless=true
        # graphics_device_id=-1
    )
    TRAIN_ARGS+=(
        # headless=false
    )
    BASE_ARGS+=(
        task.env.terrain.terrainType=plane
        task.env.randomCommandVelocityRanges.linear_x=[0.2,0.2]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]

        # "++task.env.defaultJointAngles='{joint_1_0: 0,joint_1_1: 0,joint_1_2: 0,joint_1_3: 0,joint_2_0: 0,joint_2_1: 0,joint_2_2: 0,joint_2_3: 0}'"
        # "++task.env.desiredJointAngles='{joint_1_0: 0,joint_1_1: 0,joint_1_2: 0,joint_1_3: 0,joint_2_0: 0,joint_2_1: 0,joint_2_2: 0,joint_2_3: 0}'"
        # ~task.env.defaultJointAngles
        # ~task.env.desiredJointAngles
        # "++task.env.defaultJointAngles.joint_1_0=0"
        # "++task.env.defaultJointAngles.joint_2_0=0"
        # "++task.env.defaultJointAngles.joint_1_1=0"
        # "++task.env.defaultJointAngles.joint_2_1=0"
        # "++task.env.defaultJointAngles.joint_1_2=0"
        # "++task.env.defaultJointAngles.joint_2_2=0"
        # "++task.env.defaultJointAngles.joint_1_3=0"
        # "++task.env.defaultJointAngles.joint_2_3=0"
        # "++task.env.desiredJointAngles.joint_1_0=0"
        # "++task.env.desiredJointAngles.joint_2_0=0"
        # "++task.env.desiredJointAngles.joint_1_1=0"
        # "++task.env.desiredJointAngles.joint_2_1=0"
        # "++task.env.desiredJointAngles.joint_1_2=0"
        # "++task.env.desiredJointAngles.joint_2_2=0"
        # "++task.env.desiredJointAngles.joint_1_3=0"
        # "++task.env.desiredJointAngles.joint_2_3=0"
    )
}

a1(){
    # bash run.sh a1Terrain -p
    base
    task=A1Terrain
    PLAY_ARGS+=(
        # checkpoint=../outputs/A1Terrain/False/2024-05-15_23-54-50/runs/A1Terrain_15-23-54-50/nn/A1Terrain.pth
        checkpoint=../outputs/A1Terrain/False/2024-05-16_00-50-46/runs/A1Terrain_16-00-50-46/nn/A1Terrain.pth
        task.env.dataPublisher.enable=true
        num_envs=15
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
        checkpoint=../assets/checkpoints/A1Terrain.pth
        task.env.dataPublisher.enable=true
        num_envs=15
        task.env.terrain.terrainType=plane
        task.env.terrain.numLevels=3
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
    )
}


anymalTerrain(){
    # bash run.sh anymalTerrain -p
    base
    task=AnymalTerrain
    checkpoint=../assets/checkpoints/AnymalTerrain.pth
    PLAY_ARGS+=(
        num_envs=2
        task.env.terrain.terrainType=plane
        # task.task.randomize=true
        # env.enableDebugVis=True
    )
}

base(){
    checkpoint=null
    ENTRY_POINT=train.py
    PLAY_ARGS=(
        test=true
    )
    
    TRAIN_ARGS=(
        headless=true
    )
    BASE_ARGS=()
    KEYBOARD_ARGS=(
        task.env.viewer.keyboardOperator=true
    )
}

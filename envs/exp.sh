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
        task.env.enableUDP=True
        
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
        checkpoint=../outputs/RobotDog/False/2024-05-14_15-20-17-dog3kg/runs/RobotDog_14-15-20-17/nn/RobotDog.pth
        num_envs=1
        task.env.enableUDP=true
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
        # task.env.enableUDP=True
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


a1Terrain(){
    # bash run.sh a1Terrain -p
    base
    task=A1Terrain
    checkpoint=../assets/checkpoints/A1Terrain.pth
    PLAY_ARGS+=(
        task.env.enableUDP=true
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
    KEYBOARD_ARGS=()
}

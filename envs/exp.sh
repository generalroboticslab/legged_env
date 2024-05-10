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
#         task.env.urdfAsset.file=urdf/pool_of_dog/RobotDog4/RobotDog.urdf
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




dog(){
    base
    task=RobotDog
    PLAY_ARGS+=(
        checkpoint=../assets/checkpoints/RobotDog.pth
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
        task.env.terrain.terrainType=plane
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
        num_envs=15
        # task.env.terrain.terrainType=plane
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

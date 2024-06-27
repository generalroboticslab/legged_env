# #!/bin/bash

biped(){
# remove heightmap
    base
    task=Biped
    TRAIN_ARGS+=(
        train.params.config.max_epochs=5000
        headless=false
    )
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/False/20240624_182925/runs/Biped_24-18-29-25/nn/Biped.pth
        num_envs=16
        task.env.dataPublisher.enable=true

        task.sim.dt=0.0025
        ++task.env.renderFPS=100
        # ++task.env.renderFPS=50
        task.env.learn.episodeLength_s=999
        # task.env.terrain.terrainType=plane
    )
    BASE_ARGS+=(
        task.sim.physx.num_position_iterations=2 # reduced for faster training # works!
        # dynamics matching
        ++task.env.assetDofProperties.friction=0.008
        ++task.env.assetDofProperties.armature=0.2

        ++task.env.baseHeightOffset=0.05
        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]
        ++task.env.assetDofProperties.velocity=10
        
        task.env.randomCommandVelocityRanges.linear_x=[0.5,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]
        
        task.env.terrain.terrainType=plane
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.difficultySale=0.2
        # task.env.terrain.curriculum=true
        # task.env.terrain.terrainProportions=[1,1,1,0,0,0,1,1,0,0]
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
    task.env.urdfAsset.AssetOptions.collapse_fixed_joints=false
    ++task.env.urdfAsset.root=../evolutionary_loop/assets/robogrammar_bank
    task.env.urdfAsset.file="robot_1350/robot.urdf"
    task.env.terrain.terrainType=plane
    task.env.randomCommandVelocityRanges.linear_x=[-0.5,0.5]
    task.env.randomCommandVelocityRanges.linear_y=[-0.5,0.5]
    task.env.randomCommandVelocityRanges.yaw=[-1,1]
    )
}


bug(){
    dog
    PLAY_ARGS+=(
        num_envs=1
        checkpoint=outputs/RobotDog/False/2024-05-28_00-55-58/runs/RobotDog_28-00-55-58/nn/RobotDog.pth
    )
    BASE_ARGS+=(
    # task.env.terrain.terrainType=plane
    ++task.env.urdfAsset.root="../evolutionary_loop/assets"
    task.env.urdfAsset.file="BugBot4_BugBot3_top-z-0_BugBot2_bottom-x-0/BugBot4_BugBot3_top-z-0_BugBot2_bottom-x-0.urdf"
    )
}



dog(){
    base
    task=RobotDog
    PLAY_ARGS+=(
        checkpoint=outputs/RobotDog/False/20240627_003338/runs/RobotDog_27-00-33-38/nn/RobotDog.pth
        # checkpoint=assets/checkpoints/RobotDog.pth
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
        task.env.randomCommandVelocityRanges.linear_x=[-0.5,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[-0.5,0.5]
        task.env.randomCommandVelocityRanges.yaw=[-1,1]

        task.env.terrain.terrainType=heightfield
        task.env.terrain.difficultySale=0.2
        task.env.terrain.curriculum=true
        task.env.terrain.terrainProportions=[1,1,1,0,0,0,1,1,0,0]
    )
}

a1(){
    # bash run.sh a1Terrain -p
    base
    task=A1Terrain
    PLAY_ARGS+=(
        # checkpoint=assets/checkpoints/A1Terrain.pth
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

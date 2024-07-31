# #!/bin/bash



biped_arm_passive(){
    biped_arm
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=null
    )
    TRAIN_ARGS+=(
        headless=false
    )
    BASE_ARGS+=(
        ++task.env.enablePassiveDynamics=true
    )
}

biped_arm(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_arm/runs/BipedAsymm_25-17-54-06/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/v6_vis_URDF_thin_arms/v6_vis_URDF.urdf

        ++task.env.assetDofProperties.damping=0
        ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960,0.3600,0.3600]
        ++task.env.assetDofProperties.friction=[0.0115,0.0100,0.0100,0.0800,0.3800,0.0115,0.0100,0.0100,0.0800,0.3800,0.0100,0.0100]
        task.env.control.stiffness=100
        task.env.control.damping=6 # 1.0

        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213,0,0]

        ++task.env.urdfAsset.footName=foot
    )
}


biped_passive(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_27-16-29-47/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_28-12-52-48/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-16-54-44/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-18-50-06/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-18-59-57/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-19-06-19/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-19-21-17/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-19-40-38/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-19-49-35/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-19-59-14/nn/BipedAsymm.pth # # ++task.env.learn.reward.passive_action.scale=0.1

        # good
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_29-19-58-55/nn/BipedAsymm.pth # ++task.env.learn.reward.passive_action.scale=0.05

        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_30-12-37-58/nn/BipedAsymm.pth # trained on ser02 # ++task.env.learn.reward.passive_action.scale=0.05
        
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_30-13-48-09/nn/BipedAsymm.pth  # ++task.env.learn.reward.passive_action.scale=0.02

        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_30-13-48-45/nn/BipedAsymm.pth # # ++task.env.learn.reward.passive_action.scale=0.01

        checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_30-16-34-59/nn/BipedAsymm.pth

        num_envs=8
    )
    BASE_ARGS+=(
        ++task.env.enablePassiveDynamics=true
        ++task.env.learn.reward.passive_action.scale=0.05
        # ++task.env.learn.reward.passive_action.scale=0.01
        # ++task.env.learn.reward.passive_action.scale=0.02
        # ++task.env.learn.reward.passive_action.scale=0.1

        # graphics_device_id=-1

    )
}


biped_hang(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=null
    )
    BASE_ARGS+=(

        ++task.env.baseHeightOffset=1
        task.env.urdfAsset.AssetOptions.fix_base_link=true
        # ++task.env.baseHeightOffset=0.001
        ++task.env.baseHeightOffset=0.5

        pipeline=cpu


        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false
    )
}

biped_lstm(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_lstm/runs/BipedAsymmLSTM_26-18-30-16/nn/BipedAsymmLSTM.pth
    )
    BASE_ARGS+=(
        train=BipedPPOAsymmLSTM
    )
}

biped_single_contact(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_single_contact/runs/BipedAsymm_26-19-00-56/nn/BipedAsymm.pth #bad
        # checkpoint=outputs/Biped/train/biped_single_contact/runs/BipedAsymm_26-19-07-39/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_single_contact/runs/BipedAsymm_26-19-12-49/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_single_contact/runs/BipedAsymm_27-12-39-56/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.learn.guided_contact.enable=False
        task.env.learn.reward.air_time.scale=1
        task.env.learn.reward.stance_time.scale=0
        task.env.learn.reward.single_contact.scale=0.5
        task.env.learn.reward.should_contact.scale=0
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contact,heightMap]
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
    )
}

biped_foot_height_0(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_foot_height_0/runs/BipedAsymm_30-22-31-46/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.foot_height.scale=0
    )    
}

biped(){
    base
    change_hydra_dir
    PLAY_ARGS+=(
        headless=false
        
        pipeline=cpu

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-16-05-27/nn/BipedAsymm.pth
        
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-16-10-26/nn/BipedAsymm.pth

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-16-21-20/nn/BipedAsymm.pth
       
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-17-01-13/nn/BipedAsymm.pth #dof_pow 1e-4
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-17-01-26/nn/BipedAsymm.pth #dof_pow 1e-5 # bad
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-17-31-51/nn/BipedAsymm.pth # dof_acc = -5e-7
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-17-33-56/nn/BipedAsymm.pth # foot_height = 1.0

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-17-41-53/nn/BipedAsymm.pth # foot_height.clamp_max=0.05
        
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-17-48-19/nn/BipedAsymm.pth

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-17-54-01/nn/BipedAsymm.pth # task.env.learn.reward.foot_height.scale=1.0

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-18-20-57/nn/BipedAsymm.pth
        
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-18-40-00/nn/BipedAsymm.pth

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-23-22-57/nn/BipedAsymm.pth

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_30-23-23-15/nn/BipedAsymm.pth

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_31-00-10-05/nn/BipedAsymm.pth

        checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_31-00-26-41/nn/BipedAsymm.pth

        num_envs=32
        ++task.env.viewer.follower_offset=[0.3,0.8,0.2]

        task.env.learn.episodeLength_s=999

        task.env.terrain.maxInitMapLevel=9
        # task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=2
        # task.env.terrain.curriculum=False
        task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.terrainType=plane
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,1,0]

        # task.sim.dt=0.0025
        # ++task.env.renderFPS=100
        ++task.env.renderFPS=50

        task.env.dataPublisher.enable=true

        task.env.randomize.push.enable=false
        task.env.randomize.push.interval_s=1
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false

    )
    BASE_ARGS+=(
        task=Biped
        # headless=true

        # task.env.randomize.push.interval_s=4

        train.params.config.horizon_length=32
        train.params.config.mini_epochs=8
        # train.params.config.minibatch_size=16384

        # train.params.config.gamma=0.995

        # train.params.config.max_epochs=1500

        ++task.env.assetDofProperties.velocity=10

        task.env.control.stiffness=80
        task.env.control.damping=5
        ++task.env.assetDofProperties.damping=0
        ++task.env.assetDofProperties.armature=[0.3240,0.4050,0.3240,0.3240,0.1080,0.3240,0.4050,0.3240,0.3240,0.1080]
        ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]


        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]

        task.env.randomCommandVelocityRanges.linear_x=[-0.2,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[-0.2,0.2]
        task.env.randomCommandVelocityRanges.yaw=[-1,1]

        # task.env.control.stiffness=60.0
        # task.env.control.damping=2.0 # 1.0

        task.env.control.actionScale=1.0

        task.env.randomize.friction.range=[0,1.0]

        task.env.learn.reward.orientation.scale=0.5
        task.env.learn.reward.orientation.exp_scale=-10
        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.learn.reward.dof_acc.scale=0 #-5e-7 # 0
        task.env.learn.reward.dof_vel.scale=0

        # task.env.learn.reward.dof_pos.scale=0 #-0.2 # -0.5 # -0.2
        task.env.learn.reward.dof_pos.scale=-0.2

        task.env.learn.reward.dof_pow.scale=0 #-1e-5
        task.env.learn.reward.base_height.scale=0
        task.env.learn.reward.air_time.scale=0
        task.env.learn.reward.stance_time.scale=0
        task.env.learn.reward.single_contact.scale=0
        task.env.learn.reward.contact_force.scale=0
        task.env.learn.reward.impact.scale=0
        task.env.learn.reward.stumble.scale=0
        task.env.learn.reward.collision.scale=-0.25 # 0
        task.env.learn.reward.action.scale=0
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.dof_limit.scale=0
        task.env.learn.allowKneeContacts=false
        task.env.learn.reward.should_contact.scale=1
        task.env.learn.reward.slip.scale=0
        task.env.learn.reward.foot_height.scale=0.05 # 1.0 # 1.0 # 2.0


        task.env.terrain.difficultySale=0.2
        task.env.terrain.curriculum=true
        # task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0]
        task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,1,0]
        task.env.terrain.slopeTreshold=0.05
        task.env.terrain.terrainType=trimesh
        # task.env.terrain.maxInitMapLevel=9

        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact,heightMap]
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget]
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[512,256,128]
    )
}


shoe(){
    dog
    PLAY_ARGS+=(
        checkpoint=outputs/RobotDog/False/20240613_024701/runs/RobotDog_13-02-47-01/nn/RobotDog.pth
    )
    
    BASE_ARGS+=(
    task=RobotDog
    # task.env.terrain.terrainType=plane
    ++task.env.urdfAsset.root="../evolutionary_loop/assets"
    task.env.urdfAsset.file="ShoeBot_BugBot3_top-z-0_BreadBot_bottom-x-3/ShoeBot_BugBot3_top-z-0_BreadBot_bottom-x-3.urdf"
    )
}


frog(){
    dog
    
    PLAY_ARGS+=(
        num_envs=1
        checkpoint=outputs/RobotDog/False/20240604_155930/runs/RobotDog_04-15-59-30/nn/RobotDog.pth
        # task.env.learn.episodeLength_s=5
    )
    BASE_ARGS+=(
    task=RobotDog
    # task.env.terrain.terrainType=plane
    ++task.env.urdfAsset.root="../evolutionary_loop/assets"
    task.env.urdfAsset.file="FrogBot5_FrogBot1_top-z-0_FrogBot5_bottom-x-5/FrogBot5_FrogBot1_top-z-0_FrogBot5_bottom-x-5.urdf"
    )
}


cricket(){
    dog
    
    PLAY_ARGS+=(
        num_envs=1
        checkpoint=outputs/RobotDog/False/20240603_173047/runs/RobotDog_03-17-30-47/nn/RobotDog.pth
        # task.env.learn.episodeLength_s=5
    )
    TRAIN_ARGS+=(
        # headless=false
    )
    BASE_ARGS+=(
    task=RobotDog
    # task.env.terrain.terrainType=plane
    ++task.env.urdfAsset.root="../evolutionary_loop/assets"
    task.env.urdfAsset.file="CricketBot_RobotDog_top-z-1_CanBot_bottom-x-0/CricketBot_RobotDog_top-z-1_CanBot_bottom-x-0.urdf"
    )
}



robogram(){
    base
    PLAY_ARGS+=(
        
        checkpoint=outputs/RobotDog/train/20240710_012200/runs/RobotDog_10-01-22-00/nn/RobotDog.pth
        num_envs=2
        task.env.dataPublisher.enable=true
        task.env.learn.episodeLength_s=999
    )

    BASE_ARGS+=(
    task=RobotDog
    task.env.urdfAsset.collision_filter=1
    # task.env.urdfAsset.AssetOptions.collapse_fixed_joints=true
    task.env.urdfAsset.AssetOptions.collapse_fixed_joints=true
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
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=assets/checkpoints/RobotDog.pth
        checkpoint=outputs/RobotDog/train/dog/runs/RobotDog_26-18-06-47/nn/RobotDog.pth
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
        task=RobotDog
        # ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        # ++task.env.urdfAsset.AssetOptions.vhacd_params.max_convex_hulls=3
        # ++task.env.urdfAsset.AssetOptions.vhacd_params.max_num_vertices_per_ch=16
        # task.env.terrain.terrainType=plane
        task.env.terrain.terrainType=heightfield
    )
}

a1(){
    # bash run.sh a1Terrain -p
    base
    change_hydra_dir
    PLAY_ARGS+=(
        # task.env.randomCommandVelocityRanges.linear_x=[0,0]
        # task.env.randomCommandVelocityRanges.linear_y=[0,0]
        # task.env.randomCommandVelocityRanges.yaw=[0,0]
        num_envs=2
        # checkpoint=assets/checkpoints/A1Terrain.pth
        checkpoint=outputs/A1Terrain/train/a1/runs/A1Terrain_26-17-37-01/nn/A1Terrain.pth
        task.env.dataPublisher.enable=true
    )
    BASE_ARGS+=(
        task=A1Terrain
        task.env.terrain.terrainType=plane
    )
}

a1Terrain(){
    # bash run.sh a1Terrain -p
    base
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=assets/checkpoints/A1Terrain.pth
        # checkpoint=outputs/A1Terrain/train/20240726_172127/runs/A1Terrain_26-17-21-28/nn/A1Terrain.pth
        checkpoint=outputs/A1Terrain/train/a1Terrain/runs/A1Terrain_26-17-40-15/nn/A1Terrain.pth
        # task.env.dataPublisher.enable=true
        # ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        task.env.enableDebugVis=True
        task.env.dataPublisher.enable=true
        num_envs=20
        # task.env.terrain.terrainType=heightfield
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        # task.env.terrain.difficultySale=0.5

    )
    BASE_ARGS+=(
        task=A1Terrain
    )
}


anymalTerrain(){
    # bash run.sh anymalTerrain -p
    base
    change_hydra_dir
    PLAY_ARGS+=(
        num_envs=2
        # checkpoint=assets/checkpoints/AnymalTerrain.pth
        checkpoint=outputs/AnymalTerrain/train/anymalTerrain/runs/AnymalTerrain_26-18-04-39/nn/AnymalTerrain.pth
       task.env.enableDebugVis=True
        task.env.dataPublisher.enable=true
        num_envs=20
        # task.env.terrain.terrainType=heightfield
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        # task.env.terrain.difficultySale=0.5
    )

    BASE_ARGS+=(
        task=AnymalTerrain
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
    ENTRY_POINT=dyanmics_matching.py
    change_hydra_dir
    BASE_ARGS+=(
        task=Biped

        task.env.terrain.terrainType=plane
        ++task.env.baseHeightOffset=1
        task.env.urdfAsset.AssetOptions.fix_base_link=true
        ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0]
        ++task.env.initialJointPositions=[0,0,0,0,0,0,0,0,0,0]

        task.env.learn.episodeLength_s=555

        task.env.urdfAsset.file=urdf/v6_vis_URDF_thin/v6_vis_URDF.urdf

        task.env.dataPublisher.enable=true

        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false



        task.env.control.actionScale=1
        task.env.control.decimation=4
        ++task.env.renderFPS=50
        num_envs=1

        ++task.env.assetDofProperties.velocity=20
        ++task.env.assetDofProperties.stiffness=0
        # ++tast.env.assetDofProperties.effort=200


        # task.env.control.stiffness=80
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3240,0.4050,0.3240,0.3240,0.1080,0.3240,0.4050,0.3240,0.3240,0.1080]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

        # task.env.control.stiffness=100
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3200,0.3800,0.3200,0.3200,0.0900,0.3200,0.3800,0.3200,0.3200,0.0900]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

        # task.env.control.stiffness=120
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

        # task.env.control.stiffness=140
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

        # task.env.control.stiffness=120
        # task.env.control.damping=6
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.2520,0.3150,0.2520,0.2520,0.0840,0.2520,0.3150,0.2520,0.2520,0.0840]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0100,0.0100,0.0400,0.1500,0.0115,0.0100,0.0100,0.0400,0.1500]

        # task.env.control.stiffness=120
        # task.env.control.damping=6.01
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

        task.env.control.stiffness=140
        task.env.control.damping=7
        ++task.env.assetDofProperties.damping=0
        ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
        ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]



        
        # motor_id:|  0   |  1   |  2   |  3   |  4   |
        # damping:  0.0000,0.0000,0.0000,0.0000,0.0000
        # armature: 0.1800,0.3600,0.1800,0.1800,0.0330
        # friction: 0.0000,0.0056,0.0066,0.0415,0.1000

        #values for 60kp, 1kd
        # ++task.env.assetDofProperties.damping=[0.8500,0.4200,0.6000,0.8500,0.9000,0.8500,0.4200,0.6000,0.8500,0.9000]
        # ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0050,0.0150,0.1300,0.3500,0.0115,0z.0050,0.0150,0.1300,0.3500]


        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=0
        # ++task.env.assetDofProperties.friction=0

        ##################tuning start ########################
        # for motors 0 and 5
        # ++task.env.assetDofProperties.damping=0.75
        # ++task.env.assetDofProperties.friction=0.0031
        # ++task.env.assetDofProperties.armature=0.29

        # for motors 1 and 6 and 3 and 8
        # ++task.env.assetDofProperties.damping=0.88
        # ++task.env.assetDofProperties.friction=0.00345
        # ++task.env.assetDofProperties.armature=0.25

        # # for motors 2 and 7 for ranges from 0 to pi/20
        # ++task.env.assetDofProperties.damping=3.3
        # ++task.env.assetDofProperties.friction=0.0025
        # ++task.env.assetDofProperties.armature=0.5

        # for motors 4 and 9 
        # ++task.env.assetDofProperties.damping=0.88
        # ++task.env.assetDofProperties.friction=0.2
        # ++task.env.assetDofProperties.armature=0.1
        ################## tuning end ############################



    )
}


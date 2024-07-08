# #!/bin/bash


biped_urdf_check(){
    biped_baseline
    change_hydra_dir
    TRAIN_ARGS+=(
        headless=false
    )

    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/biped_long_foot_6_2/biped_v6_terrain.urdf
    )
}





biped_stairs_trimesh_no_push(){
    biped_stairs_trimesh
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_stairs_trimesh_no_push/runs/BipedAsymm_08-13-09-11/nn/last_BipedAsymm_ep_850_rew_22.430407.pth # feet apart, bad
        checkpoint=outputs/Biped/train/biped_stairs_trimesh_no_push/runs/BipedAsymm_08-13-09-11/nn/last_BipedAsymm_ep_1250_rew_23.542065.pth
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.terrainType=trimesh
        
        # task.env.terrain.terrainType=plane

    )
    TRAIN_ARGS+=(
        headless=false
    )

    BASE_ARGS+=(        
        task.env.randomize.push.enable=false
    )
}


biped_stairs_trimesh_less_push(){
    biped_stairs_trimesh
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_stairs_trimesh_less_push/runs/BipedAsymm_08-12-25-49/nn/last_BipedAsymm_ep_4450_rew_24.30607.pth # feet apart, bad
        checkpoint=outputs/Biped/train/biped_stairs_trimesh_less_push/runs/BipedAsymm_08-12-25-49/nn/last_BipedAsymm_ep_1500_rew_23.973406.pth
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.terrainType=trimesh
        
        # task.env.terrain.terrainType=plane

    )
    TRAIN_ARGS+=(
        headless=false
    )

    BASE_ARGS+=(        
        task.env.randomize.push.interval_s=10
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
    )
}

biped_stairs_trimesh(){
    biped_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_stairs_trimesh/runs/BipedAsymm_08-12-16-37/nn/last_BipedAsymm_ep_5000_rew_24.258286.pth
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.terrainType=trimesh

    )
    TRAIN_ARGS+=(
        headless=true
    )

    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/biped_long_foot_6_2/biped_v6_terrain.urdf
        task.env.terrain.terrainProportions=[0,0,0,1,0,0,0,0,0]
        task.env.terrain.terrainType=trimesh
        task.env.terrain.slopeTreshold=0

    )
}

biped_stairs(){
    biped_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_stairs/runs/BipedAsymm_08-12-01-40/nn/last_BipedAsymm_ep_800_rew_23.057068.pth
        checkpoint=outputs/Biped/train/biped_stairs/runs/BipedAsymm_08-12-01-40/nn/last_BipedAsymm_ep_1800_rew_23.950386.pth
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.terrainType=plane
    )
    TRAIN_ARGS+=(
        headless=false
    )

    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/biped_long_foot_6_2/biped_v6_terrain.urdf
        task.env.terrain.terrainProportions=[0,0,0,1,0,0,0,0,0]
    )
}



biped_baseline(){
    # reproduce outputs/Biped/train/biped_c2r5h3e3p6
    base
    task=Biped
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_baseline/runs/BipedAsymm_07-19-11-08/nn/BipedAsymm.pth # baseHeightOffset=0.05
        # checkpoint=outputs/Biped/train/biped_baseline/runs/BipedAsymm_07-19-59-54/nn/BipedAsymm.pth # baseHeightOffset=0.1
        checkpoint=outputs/Biped/train/biped_baseline/runs/BipedAsymm_07-20-21-40/nn/BipedAsymm.pth # terrain horizontal_scale=0.1
        # task.env.terrain.terrainType=plane
        task.env.terrain.terrainType=heightfield
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.horizontalScale=0.05

        # task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0]

        # terrain types: [rough_up, rough_down, rough_flat, stair_up, stair_down, smooth_up, smooth_down, discrete, stepping_stone]
        task.env.terrain.terrainProportions=[0,0,0,1,0,0,0,0,0]

        task.env.terrain.difficultySale=0.2

        num_envs=20
        headless=false
    )

    BASE_ARGS+=(

        headless=true
        train.params.config.max_epochs=5000

        task.sim.physx.num_position_iterations=2

        task.env.urdfAsset.file=urdf/biped_long_foot/biped_v6_terrain.urdf
        ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_convex_hulls=1
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_num_vertices_per_ch=64

        ++task.env.assetDofProperties.damping=0.88
        ++task.env.assetDofProperties.friction=0.00345
        ++task.env.assetDofProperties.armature=0.25
        ++task.env.assetDofProperties.velocity=10


        # ++task.env.baseHeightOffset=0.1
        ++task.env.baseHeightOffset=0.05
        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]

        task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]

        task.env.terrain.terrainType=heightfield
        task.env.terrain.difficultySale=0.2
        task.env.terrain.curriculum=true
        task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,0,0]

        task.env.control.stiffness=60.0
        task.env.control.actionScale=1.0

        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]

        task.env.learn.reward.feetSlip.scale=-0.1
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[1024,512,256]

        task.env.randomize.push.interval_s=3
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]

        task.env.randomize.erfi.enable=true
        task.env.randomize.erfi.rfi_range=[-3.0,3.0]
        task.env.randomize.erfi.rao_range=[-3.0,3.0]

        task.env.learn.dofPositionNoise=0.02

    )
    
}

biped_terrain5(){ # 
    biped_terrain
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_terrain5/runs/BipedAsymm_02-07-54-47/nn/BipedAsymm.pth
        # task.env.terrain.terrainType=plane

        # num_envs=20
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False
        # task.env.dataPublisher.enable=false

    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.6]
        task.env.randomCommandVelocityRanges.linear_y=[-0.1,0.1]
        task.env.randomCommandVelocityRanges.yaw=[-1,1] # biped_c2r12 add rotation
    )
}


biped_terrain4(){ # 
    biped_terrain
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_terrain4/runs/BipedAsymm_02-01-49-42/nn/last_BipedAsymm_ep_5000_rew_23.48193.pth
        task.env.terrain.terrainType=plane
        # num_envs=20
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[-0.4,0.6]
        task.env.randomCommandVelocityRanges.linear_y=[-0.1,0.1]
        task.env.randomCommandVelocityRanges.yaw=[-1,1] # biped_c2r12 add rotation
    )
}

biped_terrain3(){ # 
    biped_terrain
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_terrain3/runs/BipedAsymm_02-00-21-54/nn/last_BipedAsymm_ep_5000_rew_23.475672.pth
        task.env.terrain.terrainType=plane
        # num_envs=20
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[-0.4,0.6]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[-1,1] # biped_c2r12 add rotation
    )
}
biped_terrain2(){  # somehow the step hight is low...

    biped_terrain
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_terrain2/runs/BipedAsymm_02-00-21-37/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_terrain2/runs/BipedAsymm_02-00-21-37/nn/last_BipedAsymm_ep_5000_rew_23.754187.pth
        task.env.terrain.terrainType=plane
        # num_envs=20
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[-0.4,0.6]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0] # biped_c2r12 add rotation
    )
}

biped_terrain(){ # works well, sighltly shaky
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_terrain/runs/BipedAsymm_01-23-02-14/nn/BipedAsymm.pth
        task.env.terrain.terrainType=plane
        # num_envs=20
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False
    )
    BASE_ARGS+=(
        task.env.terrain.terrainType=heightfield
        task.env.terrain.difficultySale=0.2
        task.env.terrain.curriculum=true
        task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,0,0]
    )
}

biped_m2(){ # add mass, works fine
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_m2/runs/BipedAsymm_02-00-05-59/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.randomize.baseMass.enable=True
        task.env.randomize.baseMass.range=[0,0.5]
    )
}

biped_m1(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_m1/runs/BipedAsymm_01-23-50-24/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.randomize.baseMass.enable=True
        task.env.randomize.baseMass.range=[0,1]
    )
}


biped_obs1(){ # orientation x2 , does not move at 0.1 m/s
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_obs1/runs/BipedAsymm_01-22-57-57/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.orientation.scale=-1.0
    )
}

biped_c80_1(){ # control stiffness=80,scale=1, not as good.
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c80_1/runs/BipedAsymm_01-23-43-39/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.control.stiffness=80.0
        task.env.control.actionScale=1.0
    )
}

biped_c80_2(){ # bad, stiffness may be too high
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_01-22-54-05/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.control.stiffness=80.0
        task.env.control.actionScale=2.0
    )
}

biped_no_stance(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_no_stance/runs/BipedAsymm_05-18-53-10/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.feetStanceTime.scale=0
        task.env.learn.reward.feetAirTime.scale=0.2
    )
}

biped_dyn_no_stance(){ # good 👍, BAD In real..
    biped_dyn
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_dyn_no_stance/runs/BipedAsymm_05-19-10-25/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        task.env.learn.reward.feetStanceTime.scale=0
        task.env.learn.reward.feetAirTime.scale=0.2
    )
}

biped_dyn(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_dyn/runs/BipedAsymm_05-19-01-03/nn/BipedAsymm.pth #bad
        # checkpoint=outputs/Biped/train/biped_dyn/runs/BipedAsymm_05-19-01-03/nn/last_BipedAsymm_ep_4850_rew_24.831345.pth # BAD
        checkpoint=outputs/Biped/train/biped_dyn/runs/BipedAsymm_07-18-16-43/nn/BipedAsymm.pth

        test=export
        num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        ++task.env.assetDofProperties.damping=[0.8500,0.4200,0.6000,0.8500,0.9000,0.8500,0.4200,0.6000,0.8500,0.9000]
        ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        ++task.env.assetDofProperties.friction=[0.0115,0.0050,0.0150,0.1300,0.3500,0.0115,0.0050,0.0150,0.1300,0.3500]

        task.env.urdfAsset.file=urdf/biped/biped_v6.urdf

    )
}

biped(){
    base
    change_hydra_dir
    task=Biped

    TRAIN_ARGS+=(
        train.params.config.max_epochs=5000
        headless=true
    )
    PLAY_ARGS+=(

        task.env.learn.addNoise=false
        task.env.randomize.friction.enable=false
        task.env.randomize.push.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.episodeLength_s=999

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_05-17-22-18/nn/BipedAsymm.pth #NUM 5

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_05-17-18-54/nn/BipedAsymm.pth #NUM 4

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_05-17-05-49/nn/BipedAsymm.pth # with indivisual dynamics+ 0.01 damping

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_05-16-53-21/nn/BipedAsymm.pth # old with 1 dynamics
    
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_05-16-46-14/nn/BipedAsymm.pth # new with individual dynamics

        num_envs=2
        test=export
        num_envs=1 # exported policy only works with 1 num of env
        
        # task.env.terrain.terrainType=plane
        # num_envs=20
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False

        task.env.dataPublisher.enable=true
        # task.sim.dt=0.0025
        # ++task.env.renderFPS=100
        ++task.env.renderFPS=50

        task.env.learn.episodeLength_s=999
    )
    BASE_ARGS+=(
        
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        # observation does not have linearVelocity
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.learn.reward.feetSlip.scale=-0.1
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[1024,512,256]

        # todo check if this is really needed
        task.env.control.stiffness=60.0
        task.env.control.actionScale=1.0

        task.sim.physx.num_position_iterations=2 # reduced for faster training # works!
        
        # # dynamics matching
        
        ## num 1
        ++task.env.assetDofProperties.damping=0.88
        ++task.env.assetDofProperties.friction=0.00345
        ++task.env.assetDofProperties.armature=0.25

        ## num 2
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.1800,0.3600,0.1800,0.1800,0.0330,0.1800,0.3600,0.1800,0.1800,0.0330]
        # ++task.env.assetDofProperties.friction=[0.0000,0.0056,0.0066,0.0415,0.1000,0.0000,0.0056,0.0066,0.0415,0.1000]

        ## num 3
        # ++task.env.assetDofProperties.damping=0.01
        # ++task.env.assetDofProperties.armature=[0.1800,0.3600,0.1800,0.1800,0.0330,0.1800,0.3600,0.1800,0.1800,0.0330]
        # ++task.env.assetDofProperties.friction=[0.0000,0.0056,0.0066,0.0415,0.1000,0.0000,0.0056,0.0066,0.0415,0.1000]

        # ## num 4
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.1800,0.3600,0.1800,0.1800,0.0330,0.1800,0.3600,0.1800,0.1800,0.0330]
        # ++task.env.assetDofProperties.friction=[0.0000,0.0056,0.0066,0.0415,0.0500,0.0000,0.0056,0.0066,0.0415,0.0500]


        # ## num 5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.1800,0.3600,0.1800,0.1800,0.0330,0.1800,0.3600,0.1800,0.1800,0.0330]
        # ++task.env.assetDofProperties.friction=[0.0000,0.0056,0.0066,0.0200,0.0500,0.0000,0.0056,0.0066,0.0200,0.0500]


        ++task.env.assetDofProperties.velocity=10

        ++task.env.baseHeightOffset=0.05
        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]
        
        task.env.randomize.push.interval_s=3
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
        task.env.learn.dofPositionNoise=0.02
        
        task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]

        task.env.terrain.terrainType=plane
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.difficultySale=0.2
        # task.env.terrain.curriculum=true
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,0,0]

        task.env.urdfAsset.file=urdf/biped_long_foot/biped_v6_terrain.urdf
        ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_convex_hulls=1
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_num_vertices_per_ch=64
    )
}


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

        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
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

biped_c2r13(){ # straight leg gait, warm start, terrain curriculum 0.8,bad
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
        test=export
        num_envs=1 # exported policy only works with 1 num of env
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

biped_c2r5h3e6(){ #  ERFI=6, larger critic network, very similar to biped_c2r5h3, too large, bounds loss exploded
    biped_c2r5h3
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e6/runs/BipedAsymm_01-03-27-45/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        task.env.randomize.erfi.enable=true
        task.env.randomize.erfi.rfi_range=[-6.0,6.0]
        task.env.randomize.erfi.rao_range=[-6.0,6.0]
    )
}

biped_c2r5h3e5(){ # ERFI, larger critic network, bounds loss exploded
    biped_c2r5h3
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e5/runs/BipedAsymm_01-02-56-50/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        task.env.randomize.erfi.enable=true
        task.env.randomize.erfi.rfi_range=[-5.0,5.0]
        task.env.randomize.erfi.rao_range=[-5.0,5.0]
    )
}

biped_c2r5h3e4(){ # ERFI, larger critic network
    biped_c2r5h3
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e4/runs/BipedAsymm_01-02-24-44/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        task.env.randomize.erfi.enable=true
        task.env.randomize.erfi.rfi_range=[-4.0,4.0]
        task.env.randomize.erfi.rao_range=[-4.0,4.0]
    )
}

biped_c2r5h3e3p6(){ # good 👍, higher step, baseline sim2real
    biped_c2r5h3e3
    change_hydra_dir
 PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e3p6/runs/BipedAsymm_01-22-17-13/nn/BipedAsymm.pth
        test=export
        num_envs=1 # exported policy only works with 1 num of env
        
        task.env.terrain.terrainType=plane

        # num_envs=20
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False
    )

    BASE_ARGS+=(
        task.env.randomize.push.interval_s=3
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
        task.env.learn.dofPositionNoise=0.02

        # terrain types: [rough_up, rough_down, rough_flat, stair_up, stair_down, smooth_up, smooth_down, discrete, stepping_stone]
        task.env.terrain.terrainType=heightfield
        # task.sim.physx.num_position_iterations=2 # reduced for faster training # works!
        task.env.terrain.difficultySale=0.2
        # task.env.terrain.difficultySale=0
        task.env.terrain.curriculum=true
        task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,0,0]

        task.env.urdfAsset.file=urdf/biped_long_foot/biped_v6_terrain.urdf
        ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_convex_hulls=1
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_num_vertices_per_ch=64

    )
}


biped_c2r5h3e3p5(){ # ERFI, larger critic, add baseHeight, 0 torque. 
    biped_c2r5h3e3
    change_hydra_dir

    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3p5/runs/BipedAsymm_01-17-24-17/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_c2r5h3e3p4/runs/BipedAsymm_01-15-22-03/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3p5/runs/BipedAsymm_01-18-04-07/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3p5/runs/BipedAsymm_01-18-18-03/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3p5/runs/BipedAsymm_01-18-18-03/nn/last_BipedAsymm_ep_2100_rew_22.919159.pth

        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
        
        task.env.terrain.terrainType=plane

        # num_envs=20
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False
    )
    TRAIN_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3/runs/BipedAsymm_01-10-59-50/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.randomize.push.interval_s=3
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
        task.env.learn.dofPositionNoise=0.02

        # terrain types: [rough_up, rough_down, rough_flat, stair_up, stair_down, smooth_up, smooth_down, discrete, stepping_stone]
        task.env.terrain.terrainType=heightfield
        # task.sim.physx.num_position_iterations=2 # reduced for faster training # works!

        task.env.terrain.difficultySale=0.2
        # task.env.terrain.difficultySale=0

        task.env.terrain.curriculum=true
        task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,0,0]

        task.env.urdfAsset.file=urdf/biped/biped_v6_terrain.urdf

    )
}


biped_c2r5h3e3p4(){ # ERFI, larger critic, ok
    biped_c2r5h3e3
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e3p4/runs/BipedAsymm_01-15-22-03/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3/runs/BipedAsymm_01-10-59-50/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.randomize.push.interval_s=3
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
        task.env.learn.dofPositionNoise=0.02
    )
}


biped_c2r5h3e3p3(){ # ERFI, larger critic, add baseHeight, 0 torque. bad! rotates during standing
    biped_c2r5h3e3
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e3p3/runs/BipedAsymm_01-13-16-51/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3/runs/BipedAsymm_01-10-59-50/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.randomize.push.interval_s=3
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
        task.env.learn.reward.torque.scale=-0
        task.env.learn.reward.baseHeight.scale=-0.05
    )
}


biped_c2r5h3e3p2(){ # ERFI, larger critic, 0 torque reward, bad! lots of feet slip
    biped_c2r5h3e3
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e3p2/runs/BipedAsymm_01-12-45-04/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3/runs/BipedAsymm_01-10-59-50/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.randomize.push.interval_s=3
        task.env.randomize.push.velMin=[-0.3,-0.3,-0.3,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0.3,0.3,0.3,0.3]
        task.env.learn.reward.torque.scale=-0
    )
}

biped_c2r5h3e3p1(){ # ERFI, larger critic network, feet sliping.
    biped_c2r5h3e3
    change_hydra_dir

    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3p1/runs/BipedAsymm_01-12-03-48/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_c2r5h3e3p1/runs/BipedAsymm_01-12-38-59/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3/runs/BipedAsymm_01-10-59-50/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.randomize.push.interval_s=3
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
    )
}

biped_c2r5h3e3(){ # ERFI, larger critic network, good!
    biped_c2r5h3
    change_hydra_dir

    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_c2r5h3e3/runs/BipedAsymm_01-02-14-48/nn/BipedAsymm.pth # rfi every dt
        checkpoint=outputs/Biped/train/biped_c2r5h3e3/runs/BipedAsymm_01-10-59-50/nn/BipedAsymm.pth
        
        test=export
        num_envs=1 # exported policy only works with 1 num of env


        # ++task.env.baseHeightOffset=1
        # task.env.urdfAsset.AssetOptions.fix_base_link=true

        num_envs=1
        task.env.learn.addNoise=false
        task.env.randomize.friction.enable=false
        task.env.randomize.push.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.episodeLength_s=999

    )
    TRAIN_ARGS+=(
        task.env.randomize.erfi.enable=true
        task.env.randomize.erfi.rfi_range=[-3.0,3.0]
        task.env.randomize.erfi.rao_range=[-3.0,3.0]
    )
}

biped_c2r5h3e2(){ # ERFI, larger critic network
    biped_c2r5h3
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e2/runs/BipedAsymm_01-03-20-57/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        task.env.randomize.erfi.enable=true
        task.env.randomize.erfi.rfi_range=[-2.0,2.0]
        task.env.randomize.erfi.rao_range=[-2.0,2.0]
    )
}

biped_c2r5h3e1(){ # ERFI, larger critic network
    biped_c2r5h3
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3e1/runs/BipedAsymm_01-02-23-22/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    TRAIN_ARGS+=(
        task.env.randomize.erfi.enable=true
        task.env.randomize.erfi.rfi_range=[-1.0,1.0]
        task.env.randomize.erfi.rao_range=[-1.0,1.0]
    )
}

biped_c2r5h3(){ # larger critic network, straight knee walking 👍
    biped_c2r5
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h3/runs/BipedAsymm_30-11-21-22/nn/BipedAsymm.pth # new
        # checkpoint=outputs/Biped/train/biped_c2r5h3/runs/BipedAsymm_30-11-21-22/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[1024,512,256]
    )
}

biped_c2r5h2(){ # smaller actor network,similar to biped_c2r5, more bend, feet draging
    biped_c2r5
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h2/runs/BipedAsymm_30-11-20-36/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        train.params.network.mlp.units=[256,128,64] # smaller network
        train.params.config.central_value_config.network.mlp.units=[512,256,128] # same
    )
}


biped_c2r5h1(){ # stand but not walk, bad. smaller networks does not work well
    biped_c2r5
    change_hydra_dir

    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_c2r5h1/runs/BipedAsymm_30-11-20-22/nn/BipedAsymm.pth
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env
    )
    BASE_ARGS+=(
        train.params.network.mlp.units=[256,128,64]
        train.params.config.central_value_config.network.mlp.units=[256,128,64]
    )
}


biped_c2r5(){ # can walk in small steps. use this as baseline 👍, asymmetric obs, feetSlip=-0.1
    biped_c2
    change_hydra_dir
    PLAY_ARGS+=(

        test=export
        num_envs=1 # exported policy only works with 1 num of env
        
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

# biped(){
#     # remove heightmap
#     base
#     task=Biped
#     TRAIN_ARGS+=(
#         train.params.config.max_epochs=5000
#         headless=true
#     )
#     PLAY_ARGS+=(

#         num_envs=2

#         # test=export
#         # num_envs=1 # exported policy only works with 1 num of env

#         checkpoint=outputs/Biped/train/biped/runs/Biped_29-18-50-06/nn/Biped.pth # new enlarged cmd x
        
#         task.env.dataPublisher.enable=true

#         task.sim.dt=0.0025
#         ++task.env.renderFPS=100

#         # ++task.env.renderFPS=50

#         task.env.learn.episodeLength_s=999
#         # task.env.terrain.terrainType=plane
#     )
#     BASE_ARGS+=(
#         task.sim.physx.num_position_iterations=2 # reduced for faster training # works!
#         # # dynamics matching
#         # ++task.env.assetDofProperties.friction=0.008
#         # ++task.env.assetDofProperties.armature=0.2

#         ++task.env.assetDofProperties.damping=0.88
#         ++task.env.assetDofProperties.friction=0.00345
#         ++task.env.assetDofProperties.armature=0.25
#         ++task.env.assetDofProperties.velocity=10

#         ++task.env.baseHeightOffset=0.05
#         ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]
        
        
#         # task.env.randomCommandVelocityRanges.linear_x=[0.5,0.5]
#         task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.5]
#         task.env.randomCommandVelocityRanges.linear_y=[0,0]
#         task.env.randomCommandVelocityRanges.yaw=[0,0]

#         task.env.terrain.terrainType=plane
#         # task.env.terrain.terrainType=heightfield
#         # task.env.terrain.difficultySale=0.2
#         # task.env.terrain.curriculum=true
#         # task.env.terrain.terrainProportions=[1,1,1,0,0,0,1,1,0,0]

#         "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
#     )
# }


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
        # task.env.randomCommandVelocityRanges.linear_x=[0,0]
        # task.env.randomCommandVelocityRanges.linear_y=[0,0]
        # task.env.randomCommandVelocityRanges.yaw=[0,0]
        num_envs=2
        # checkpoint=assets/checkpoints/A1Terrain.pth
        # checkpoint=outputs/A1Terrain/train/20240701_003303/runs/A1Terrain_01-00-33-03/nn/A1Terrain.pth
        # checkpoint=outputs/A1Terrain/train/20240701_004744/runs/A1Terrain_01-00-47-44/nn/A1Terrain.pth
        # checkpoint=outputs/A1Terrain/train/20240701_005534/runs/A1Terrain_01-00-55-35/nn/A1Terrain.pth
        # checkpoint=outputs/A1Terrain/train/20240701_010308/runs/A1Terrain_01-01-03-08/nn/A1Terrain.pth
        checkpoint=outputs/A1Terrain/train/20240701_012708/runs/A1Terrain_01-01-27-08/nn/A1Terrain.pth
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
        # checkpoint=assets/checkpoints/A1Terrain.pth
        checkpoint=outputs/A1Terrain/train/20240701_012708/runs/A1Terrain_01-01-27-08/nn/A1Terrain.pth
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


        task.env.control.stiffness=60
        task.env.control.damping=2
        task.env.control.actionScale=1

        task.env.control.decimation=2
        ++task.env.renderFPS=100
        num_envs=1

        ++task.env.assetDofProperties.velocity=20
        ++task.env.assetDofProperties.stiffness=0
        ++tast.env.assetDofProperties.effort=200        

        # ++task.env.assetDofProperties.damping=0.275
        # ++task.env.assetDofProperties.friction=0.0055
        # ++task.env.assetDofProperties.armature=0.2
        
        # motor_id:|  0   |  1   |  2   |  3   |  4   |
        # damping:  0.0000,0.0000,0.0000,0.0000,0.0000
        # armature: 0.1800,0.3600,0.1800,0.1800,0.0330
        # friction: 0.0000,0.0056,0.0066,0.0415,0.1000

        ++task.env.assetDofProperties.damping=[0.8500,0.4200,0.6000,0.8500,0.9000,0.8500,0.4200,0.6000,0.8500,0.9000]
        ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        ++task.env.assetDofProperties.friction=[0.0115,0.0050,0.0150,0.1300,0.3500,0.0115,0.0050,0.0150,0.1300,0.3500]

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
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false

        task.env.terrain.terrainType=plane
        ++task.env.baseHeightOffset=1
        task.env.urdfAsset.AssetOptions.fix_base_link=true
        ++task.env.defaultJointPositions=[0,1.5708,0,0,0,0.000,-1.5708,0,0,0]

        task.env.control.stiffness=0
        task.env.control.damping=0

        pipeline="gpu"

        ++task.env.assetDofProperties.velocity=20
        ++task.env.assetDofProperties.stiffness=0
        ++task.env.assetDofProperties.damping=0
        ++task.env.assetDofProperties.friction=0.008
        ++task.env.assetDofProperties.armature=0.2

        ++task.env.renderFPS=50
        task.env.learn.episodeLength_s=555
    )
}


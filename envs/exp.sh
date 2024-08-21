# #!/bin/bash



biped_arm_passive(){
    biped_arm
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=null
    )
    TRAIN_ARGS+=(
        # headless=false
    )
    BASE_ARGS+=(
        ++task.env.enablePassiveDynamics=true
        ++task.env.learn.reward.passive_action.scale=0.01 # 0.002        
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

        task.env.control.stiffness=80
        task.env.control.damping=5
        ++task.env.assetDofProperties.damping=0
        ++task.env.assetDofProperties.armature=[0.3240,0.4050,0.3240,0.3240,0.1080,0.3240,0.4050,0.3240,0.3240,0.1080,0.4050,0.4050]
        ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500,0.0200,0.0200]

        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213,0,0]

        ++task.env.urdfAsset.footName=foot
    )
}

biped_passive_run(){
    biped_passive
    _run
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_passive_run/runs/BipedAsymm_04-21-38-37/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_passive_run/runs/BipedAsymm_06-03-52-28/nn/BipedAsymm.pth # good
        # task.env.terrain.terrainType=trimesh
    )   

    BASE_ARGS+=(
    )
}


biped_run(){
    biped
    _run
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_run/runs/BipedAsymm_06-12-37-58/nn/BipedAsymm.pth
    )   

    BASE_ARGS+=(
    )
}



_run(){
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.5,1.0]
        task.env.learn.guided_contact.phase_stance_ratio=0.38
        task.env.learn.guided_contact.phase_freq=1.6666666666666667
        task.env.learn.reward.ang_vel.scale=1 #0.5
    )
}

biped_passive_jump(){
    biped_passive
    _jump
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_passive_jump/runs/BipedAsymm_06-00-52-50/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive_jump/runs/BipedAsymm_06-01-21-21/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_passive_jump/runs/BipedAsymm_06-03-25-01/nn/BipedAsymm.pth
    )
}

biped_passive(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_02-16-56-25/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_02-17-15-20/nn/BipedAsymm.pth #  task.env.learn.guided_contact.phase_stance_ratio=0.54 
        # num_envs=8
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_02-17-47-17/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_04-16-47-09/nn/BipedAsymm.pth # good.

        # task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,1,0]
        task.env.terrain.terrainProportions=[0,0,1,1,0,0,0,1,0]
        task.env.terrain.numTerrains=3
        task.env.terrain.numLevels=4
        task.env.terrain.maxInitMapLevel=3

        task.env.randomize.baseMass.enable=True
        task.env.randomize.baseMass.range=[5,5]
        
    )
    TRAIN_ARGS+=(
        # headless=false
    )
    BASE_ARGS+=(
        ++task.env.enablePassiveDynamics=true
        ++task.env.learn.reward.passive_action.scale=0.01 # 0.002

        # task.env.learn.guided_contact.phase_stance_ratio=0.54
        # task.env.learn.guided_contact.phase_freq=1.6666666666666667


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
        # checkpoint=outputs/Biped/train/biped_lstm/runs/BipedAsymmLSTM_26-18-30-16/nn/BipedAsymmLSTM.pth
        checkpoint=outputs/Biped/train/biped_lstm/runs/BipedAsymmLSTM_26-18-30-16/nn/BipedAsymmLSTM.pth # tiptoe walking :(
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

biped_dof_debug(){
    biped_debug
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_dof_debug/runs/BipedAsymm_04-13-50-05/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_dof_debug/runs/BipedAsymm_04-13-53-41/nn/BipedAsymm.pth # task.env.terrain.difficultySale=0.3
        # checkpoint=outputs/Biped/train/biped_dof_debug/runs/BipedAsymm_04-14-12-51/nn/BipedAsymm.pth # task.env.terrain.difficultySale=0.4 -> bent knee walking
        # task.env.terrain.terrainType=trimesh

    )
    BASE_ARGS+=(
        ++task.env.desiredJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]
        ++task.env.defaultJointPositions=[-0.26,0,0,0.79,0.08,0.26,0,0,-0.79,-0.08] # (self.dof_props['upper']+ self.dof_props["lower"])/2
        # task.env.terrain.difficultySale=0.3
        # task.env.terrain.difficultySale=0.4
    )
}

biped_debug(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # task.env.data_root_label=baseline

        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_04-14-44-48/nn/BipedAsymm.pth # good, but too much vibration in real

        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_04-16-04-53/nn/BipedAsymm.pth # #action_rate.scale=0.05 ->  super bent knee walking

        checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_04-16-44-07/nn/BipedAsymm.pth # task.env.learn.reward.dof_pow.scale=-1e-5 

        # task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,1,0]
        task.env.terrain.terrainProportions=[0,0,1,1,0,0,0,1,0]
        task.env.terrain.numTerrains=3
        task.env.terrain.numLevels=4
        task.env.terrain.maxInitMapLevel=3


        # num_envs=24

        # test=export
        # num_envs=1 # exported policy only works with 1 num of env

        # checkpoint=null
        # ++task.env.baseHeightOffset=1
        # task.env.urdfAsset.AssetOptions.fix_base_link=true
        # # ++task.env.baseHeightOffset=0.001
        # ++task.env.baseHeightOffset=0.5
        # pipeline=cpu


    )
    BASE_ARGS+=(
        # task.env.learn.reward.action_rate.scale=0.5

        # task.env.learn.reward.action_rate.scale=0 # 0.2
        # task.env.learn.reward.action_rate.exp_scale=-0.001

        # task.env.learn.reward.dof_acc.scale=0.2
        # task.env.learn.reward.dof_acc.exp_scale=-0.0001

        # task.env.learn.reward.impact.scale=0.2
        # task.env.learn.reward.impact.exp_scale=-0.00001

        # task.env.learn.reward.impact.scale=0.1
        # task.env.learn.reward.impact.exp_scale=-0.00001

        # task.env.learn.reward.collision.scale=0 # 0

        # task.env.learn.reward.foot_forward.scale=-0.2


        # task.env.learn.reward.dof_acc.scale=0.1
        # task.env.learn.reward.dof_acc.exp_scale=-0.0001      
        # task.env.learn.reward.dof_pos.scale=-0.02 #-0.02 #-0.01

        task.env.learn.reward.dof_pow.scale=-1e-5 #-1e-5

        # task.env.learn.reward.action_rate.scale=0.05 # 0.01 #0.05 #0.1 # 0 # 0.2
        # task.env.learn.reward.action_rate.exp_scale=-0.001

    )
}

_jump(){
    BASE_ARGS+=(
        task.env.terrain.terrainType=plane
        task.env.learn.guided_contact.phase_freq=1.25
        task.env.learn.guided_contact.phase_offset=[0,0]
        task.env.learn.guided_contact.phase_stance_ratio=0.5
        task.env.learn.reward.lin_vel.normalize_by=[1,1,0]

        task.env.randomCommandVelocityRanges.linear_x=[0.3,0.3]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]

        # turn off friction randomization for jumping
        task.env.randomize.friction.enable=false

        # # foot can bent more
        task.env.learn.reward.foot_orientation.scale=0.2 #0.1
        # task.env.learn.reward.foot_orientation.exp_scale=-5 # -5

        task.env.learn.reward.lin_vel.scale=0.5 # half the default
        task.env.learn.reward.foot_height.scale=1 # 10x the default
        task.env.learn.reward.foot_height.clamp_max=0.5 #0.2 # 0.1

        task.env.learn.reward.dof_acc.scale=0
        task.env.learn.reward.dof_acc.exp_scale=-1e-5

        task.env.learn.reward.dof_pos.scale=0 #-0.02

        task.env.learn.reward.foot_pos.scale=1.0
        task.env.learn.reward.foot_pos.exp_scale=-10


    )
}
biped_jump(){
    biped
    _jump
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_jump/runs/BipedAsymm_06-01-05-41/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_jump/runs/BipedAsymm_06-01-46-58/nn/BipedAsymm.pth # ok
    )
    BASE_ARGS+=(

    )
}

biped(){
    base
    change_hydra_dir
    PLAY_ARGS+=(
        headless=false
        # pipeline=cpu
        checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_05-13-32-17/nn/BipedAsymm.pth # baseline
        checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_05-17-00-22/nn/BipedAsymm.pth # new urdf
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_06-00-24-42/nn/BipedAsymm.pth #        task.env.learn.reward.foot_pos.scale=0.1,task.env.learn.reward.foot_pos.exp_scale=-5

        checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_06-02-14-09/nn/BipedAsymm.pth # 3000 epochs good. /thumbsup

        checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_06-13-51-18/nn/BipedAsymm.pth # 3000 epochs. task.env.control.stiffness=60, task.env.control.damping=6

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_06-14-22-35/nn/BipedAsymm.pth  # task.env.control.stiffness=60, task.env.control.damping=5

        num_envs=24
        ++task.env.viewer.follower_offset=[0.3,0.8,0.2]


        # test=export
        num_envs=1 # exported policy only works with 1 num of env
        
        task.env.learn.episodeLength_s=999

        task.env.terrain.maxInitMapLevel=9
        # task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=2
        # task.env.terrain.curriculum=False
        task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainType=heightfield
        task.env.terrain.terrainType=plane
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,1,0]

        # task.sim.dt=0.0025
        # ++task.env.renderFPS=100
        ++task.env.renderFPS=50

        task.env.dataPublisher.enable=true

        task.env.randomize.push.enable=false
        # task.env.randomize.push.enable=true

        task.env.randomize.push.velMin=[0.5,0,0,0,0,0]
        task.env.randomize.push.velMax=[0.5,0,0,0,0,0]
        task.env.randomize.push.interval_s=3

        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false

    )
    BASE_ARGS+=(
        task=Biped
        # headless=true

        # task.env.learn.guided_contact.phase_stance_ratio=0.54 # bad..

        # task.sim.physx.num_position_iterations=4

        # task.env.randomize.push.interval_s=4

        # train.params.config.save_best_after=200

        train.params.config.horizon_length=32
        train.params.config.mini_epochs=8
        # train.params.config.minibatch_size=16384

        # train.params.config.gamma=0.995

        train.params.config.max_epochs=3000

        task.env.urdfAsset.file=urdf/v6_vis_URDF_new_body/v6_vis_URDF.urdf


        ++task.env.assetDofProperties.velocity=10

        # task.env.control.stiffness=80
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3240,0.4050,0.3240,0.3240,0.1080,0.3240,0.4050,0.3240,0.3240,0.1080]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

        # task.env.control.stiffness=60
        # task.env.control.damping=6
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        # ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]

        task.env.control.stiffness=60
        task.env.control.damping=5
        ++task.env.assetDofProperties.damping=0
        ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]

        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]

        # task.env.randomCommandVelocityRanges.linear_x=[-0.2,0.5]
        task.env.randomCommandVelocityRanges.linear_x=[-0.2,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[-0.2,0.2]
        # task.env.randomCommandVelocityRanges.yaw=[-1,1]
        task.env.randomCommandVelocityRanges.yaw=[-0.8,0.8]

        # task.env.control.stiffness=60.0
        # task.env.control.damping=2.0 # 1.0

        # task.env.control.actionScale=1.0

        task.env.randomize.friction.range=[0,1.0]

        task.env.learn.allowKneeContacts=false
        task.env.learn.reward.lin_vel.normalize_by=[1,1,0.1]
        # task.env.learn.reward.ang_vel.scale=1 #0.5
        
        task.env.learn.reward.orientation.scale=0.5
        task.env.learn.reward.orientation.exp_scale=-10

        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.learn.reward.dof_force_target.exp_scale=-1.0

        task.env.learn.reward.dof_acc.scale=0.1
        task.env.learn.reward.dof_acc.exp_scale=-0.0001   
        task.env.learn.reward.dof_vel.scale=0
        task.env.learn.reward.dof_pos.scale=-0.02 #-0.02 #-0.01
        task.env.learn.reward.dof_pow.scale=0 #-2e-5 #-1e-5 # 0
        task.env.learn.reward.base_height.scale=0
        task.env.learn.reward.air_time.scale=0
        task.env.learn.reward.stance_time.scale=0
        task.env.learn.reward.single_contact.scale=0
        task.env.learn.reward.contact_force.scale=0
        task.env.learn.reward.impact.scale=0
        task.env.learn.reward.stumble.scale=0
        task.env.learn.reward.collision.scale=0 # -0.25 # 0
        task.env.learn.reward.action.scale=0

        task.env.learn.reward.foot_pos.scale=0.1
        task.env.learn.reward.foot_pos.exp_scale=-5

        task.env.learn.reward.foot_forward.scale=0.1
        task.env.learn.reward.foot_forward.exp_scale=-5 # -10

        task.env.learn.reward.foot_orientation.scale=0.1
        task.env.learn.reward.foot_orientation.exp_scale=-5 #-10

        task.env.learn.reward.action_rate.scale=0 # 0.01 #0.05 #0.1 # 0 # 0.2
        task.env.learn.reward.action_rate.exp_scale=-0.001

        task.env.learn.reward.dof_limit.scale=0
        task.env.learn.reward.should_contact.scale=1.0 # 0.5
        task.env.learn.reward.slip.scale=0

        # task.env.learn.reward.foot_height.scale=0.05 # 1.0 # 1.0 # 2.0
        task.env.learn.reward.foot_height.scale=0.1

        task.env.terrain.difficultySale=0.3 #0.2
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

# match_dyn(){
#     ENTRY_POINT=dyanmics_matching.py
#     change_hydra_dir
#     BASE_ARGS+=(
#         task=Biped

#         task.env.terrain.terrainType=plane
#         ++task.env.baseHeightOffset=1
#         task.env.urdfAsset.AssetOptions.fix_base_link=true
#         ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0]
#         ++task.env.initialJointPositions=[0,0,0,0,0,0,0,0,0,0]

#         task.env.learn.episodeLength_s=555

#         task.env.urdfAsset.file=urdf/v6_vis_URDF_thin/v6_vis_URDF.urdf

#         task.env.dataPublisher.enable=true

#         task.env.randomize.push.enable=false
#         task.env.randomize.friction.enable=false
#         task.env.randomize.initDofPos.enable=false
#         task.env.randomize.initDofVel.enable=false
#         task.env.randomize.erfi.enable=false
#         task.env.learn.addNoise=false



#         task.env.control.actionScale=1
#         task.env.control.decimation=4
#         ++task.env.renderFPS=50
#         num_envs=1

#         ++task.env.assetDofProperties.velocity=20
#         ++task.env.assetDofProperties.stiffness=0
#         # ++tast.env.assetDofProperties.effort=200

#         #----------------------------------------------------------------------------------------------
#         #Aug 6. dynamics rematched
#         task.env.control.stiffness=60
#         task.env.control.damping=5
#         ++task.env.assetDofProperties.damping=0
#         ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
#         ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]
        
#         #----------------------------------------------------------------------------------------------

#         # task.env.control.stiffness=80
#         # task.env.control.damping=5
#         # ++task.env.assetDofProperties.damping=0
#         # ++task.env.assetDofProperties.armature=[0.3240,0.4050,0.3240,0.3240,0.1080,0.3240,0.4050,0.3240,0.3240,0.1080]
#         # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

#         # task.env.control.stiffness=100
#         # task.env.control.damping=5
#         # ++task.env.assetDofProperties.damping=0
#         # ++task.env.assetDofProperties.armature=[0.3200,0.3800,0.3200,0.3200,0.0900,0.3200,0.3800,0.3200,0.3200,0.0900]
#         # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

#         # task.env.control.stiffness=120
#         # task.env.control.damping=5
#         # ++task.env.assetDofProperties.damping=0
#         # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
#         # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

#         # task.env.control.stiffness=140
#         # task.env.control.damping=5
#         # ++task.env.assetDofProperties.damping=0
#         # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
#         # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

#         # task.env.control.stiffness=120
#         # task.env.control.damping=6
#         # ++task.env.assetDofProperties.damping=0
#         # ++task.env.assetDofProperties.armature=[0.2520,0.3150,0.2520,0.2520,0.0840,0.2520,0.3150,0.2520,0.2520,0.0840]
#         # ++task.env.assetDofProperties.friction=[0.0115,0.0100,0.0100,0.0400,0.1500,0.0115,0.0100,0.0100,0.0400,0.1500]

#         # task.env.control.stiffness=120
#         # task.env.control.damping=6.01
#         # ++task.env.assetDofProperties.damping=0
#         # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
#         # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]

#         # task.env.control.stiffness=140
#         # task.env.control.damping=7
#         # ++task.env.assetDofProperties.damping=0
#         # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
#         # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]



        
#         # motor_id:|  0   |  1   |  2   |  3   |  4   |
#         # damping:  0.0000,0.0000,0.0000,0.0000,0.0000
#         # armature: 0.1800,0.3600,0.1800,0.1800,0.0330
#         # friction: 0.0000,0.0056,0.0066,0.0415,0.1000

#         #values for 60kp, 1kd
#         # ++task.env.assetDofProperties.damping=[0.8500,0.4200,0.6000,0.8500,0.9000,0.8500,0.4200,0.6000,0.8500,0.9000]
#         # ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
#         # ++task.env.assetDofProperties.friction=[0.0115,0.0050,0.0150,0.1300,0.3500,0.0115,0z.0050,0.0150,0.1300,0.3500]


#         # ++task.env.assetDofProperties.damping=0
#         # ++task.env.assetDofProperties.armature=0
#         # ++task.env.assetDofProperties.friction=0

#         ##################tuning start ########################
#         # for motors 0 and 5
#         # ++task.env.assetDofProperties.damping=0.75
#         # ++task.env.assetDofProperties.friction=0.0031
#         # ++task.env.assetDofProperties.armature=0.29

#         # for motors 1 and 6 and 3 and 8
#         # ++task.env.assetDofProperties.damping=0.88
#         # ++task.env.assetDofProperties.friction=0.00345
#         # ++task.env.assetDofProperties.armature=0.25

#         # # for motors 2 and 7 for ranges from 0 to pi/20
#         # ++task.env.assetDofProperties.damping=3.3
#         # ++task.env.assetDofProperties.friction=0.0025
#         # ++task.env.assetDofProperties.armature=0.5

#         # for motors 4 and 9 
#         # ++task.env.assetDofProperties.damping=0.88
#         # ++task.env.assetDofProperties.friction=0.2
#         # ++task.env.assetDofProperties.armature=0.1
#         ################## tuning end ############################



#     )
# }


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
        ++task.env.learn.enablePassiveDynamics=true
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

        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_15-15-42-36/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_16-00-15-02/nn/BipedAsymm.pth # good behaviour, but not much passive action

        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_16-12-13-20/nn/BipedAsymm.pth # passive_action.scale=0.02

        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_16-13-36-50/nn/BipedAsymm.pth

        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_17-12-22-10/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_18-12-02-19/nn/BipedAsymm.pth # sigmoid on passive action
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_18-12-29-55/nn/BipedAsymm.pth # sigmoid 10

        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_18-14-23-58/nn/BipedAsymm.pth # ++task.env.learn.reward.passive_action.scale=0.005
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_18-14-24-46/nn/BipedAsymm.pth # ++task.env.learn.reward.passive_action.scale=0
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_18-15-33-57/nn/BipedAsymm.pth # sigmoid 5 # too weak..

        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_18-16-25-54/nn/BipedAsymm.pth # 100 Hz control

        # eward.passive_action.scale=0
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-12-05-47/nn/last_BipedAsymm_ep_1350_rew_61.527977.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-12-05-47/nn/last_BipedAsymm_ep_1700_rew_62.77962.pth
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-12-05-47/nn/BipedAsymm.pth # passive curriculum 1->0
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-12-34-47/nn/BipedAsymm.pth # passive curriculum 0.8->0
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-12-52-21/nn/BipedAsymm.pth # passive curriculum 0.5->0
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-12-52-21/nn/last_BipedAsymm_ep_1950_rew_65.11146.pth # passive curriculum 0.5->0

        # ++task.env.learn.reward.passive_action.scale=0.005
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-13-37-27/nn/last_BipedAsymm_ep_1300_rew_63.802437.pth

        # ++task.env.learn.reward.passive_action.scale=0.001
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-15-30-19/nn/BipedAsymm.pth


        # task.sim.dt=0.0025
        # task.env.control.decimation=4
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-18-39-38/nn/BipedAsymm.pth

        # task.sim.dt=0.001
        # task.env.control.decimation=10
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-20-45-30/nn/BipedAsymm.pth

        # task.sim.dt=0.001
        # task.env.control.decimation=10
        # ++task.env.learn.passiveCurriculum=false
        # checkpoint=outputs/Biped/train/biped_passive/runs/BipedAsymm_19-21-29-36/nn/BipedAsymm.pth # weird behaviour



        num_envs=1
        # test=export

        # task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,1,0]
        task.env.terrain.terrainProportions=[0,0,1,1,0,0,0,1,0]
        task.env.terrain.numTerrains=3
        task.env.terrain.numLevels=4
        task.env.terrain.maxInitMapLevel=3

        # task.env.randomize.baseMass.enable=True
        # task.env.randomize.baseMass.range=[5,5]
        
        # ++task.env.renderFPS=100
    )
    TRAIN_ARGS+=(
        # headless=false
    )
    BASE_ARGS+=(

        task.sim.dt=0.005 # 200 Hz
        task.env.control.decimation=4 # 200/4 = 50 Hz
        ++task.env.renderFPS=50

        # task.sim.dt=0.0025
        # task.env.control.decimation=4

        # task.sim.dt=0.001
        # task.env.control.decimation=10

        # task.sim.dt=0.005
        # task.env.control.decimation=2
        # ++task.env.renderFPS=100

        ++task.env.learn.passiveCurriculum=false #true
        ++task.env.learn.enablePassiveDynamics=true
        ++task.env.learn.reward.passive_action.scale=0.001 #0.005 # 0 # 0.005 #0.02 # 0.01 # 0.002

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

biped_noise_kp60kd4(){
    biped_noise
    change_hydra_dir
    PLAY_ARGS+=(
        # asymetric gait, one is strait knee, another is bent knee
        checkpoint=outputs/Biped/train/biped_noise_kp60kd4/runs/BipedAsymm_07-00-43-59/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.control.stiffness=60
        task.env.control.damping=4
    )
}
biped_noise_kp60kd2(){
    biped_noise
    change_hydra_dir
    PLAY_ARGS+=(
        # asymetric gait, one is strait knee, another is bent knee
        checkpoint=outputs/Biped/train/biped_noise_kp60kd2/runs/BipedAsymm_07-00-43-39/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.control.stiffness=60
        task.env.control.damping=2
    )
}



biped_kp60kd4(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # asymetric gait, one is strait knee, another is bent knee
        # same with kd1, and kd2
        checkpoint=outputs/Biped/train/biped_kp60kd4/runs/BipedAsymm_07-00-49-10/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.control.stiffness=60
        task.env.control.damping=4
    )
}



biped_noise(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # asymetric gait, straight knee on one side
        checkpoint=outputs/Biped/train/biped_noise/runs/BipedAsymm_07-00-40-21/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        # ++task.env.max_observation_delay_steps=2
        task.env.learn.dofVelocityNoise=3.0 # 1.5
        task.env.learn.dofPositionNoise=0.02 # 0.01
    )
}



biped_delayed_3(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # very stright knee walking, small step height, not using knee joint
        checkpoint=outputs/Biped/train/biped_delayed_3/runs/BipedAsymm_07-00-33-54/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        ++task.env.max_observation_delay_steps=3 # 1 # 0 observation delay
    )
}

biped_p(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=null
        # ++task.env.baseHeightOffset=0.2
        # task.env.urdfAsset.AssetOptions.fix_base_link=true
        # task.env.control.actionScale=0 # HACK no action

        checkpoint=outputs/Biped/train/biped_p/runs/BipedAsymm_15-00-33-52/nn/BipedAsymm.pth

        # pipeline=cpu
    )
    BASE_ARGS+=(
        task.sim.physx.num_position_iterations=4
        task.sim.physx.num_velocity_iterations=2

        task.sim.dt=0.00333333333333333333
        task.env.control.decimation=6

        # task.sim.dt=0.0025
        # task.env.control.decimation=8

        # ++task.env.renderFPS=100
        

        # Aug 8. dynamics rematched
        task.env.control.stiffness=60
        task.env.control.damping=5
        ++task.env.assetDofProperties.damping=0
        #                                      #0 hipz,1 hipy,2 hipx,3 knee,4 ankl,5      6      0 hipz,1 hipy,2 hipx,3 knee,4 ankl,5      6     
        ++task.env.assetDofProperties.armature=[0.2100,0.4200,0.3800,0.2100,0.0750,0.0000,0.0000,0.2100,0.4200,0.3800,0.2100,0.0750,0.0000,0.0000]
        ++task.env.assetDofProperties.friction=[0.0040,0.0100,0.0040,0.0400,0.0200,0.0020,0.0020,0.0040,0.0100,0.0040,0.0400,0.0200,0.0020,0.0020]


        ++task.env.urdfAsset.marker_pair_names=[L_MARKER_A,L_MARKER_B,R_MARKER_A,R_MARKER_B]
        ++task.env.urdfAsset.marker_pair_length=[0.14,0.235,0.14,0.235]

        # ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,0,-0.213,0,0,0,0.000,-0.175,0.000,-0.387,0,0.213,0,0,0]
        ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        task.env.urdfAsset.file=urdf/v6biped_urdf_v2/v6biped_urdf_v2.urdf
        task.env.urdfAsset.AssetOptions.collapse_fixed_joints=false


    )


}

biped_debug(){
    biped
    change_hydra_dir
    PLAY_ARGS+=(
        # task.env.data_root_label=baseline

        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_04-14-44-48/nn/BipedAsymm.pth # good, but too much vibration in real

        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_04-16-04-53/nn/BipedAsymm.pth # #action_rate.scale=0.05 ->  super bent knee walking

        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_04-16-44-07/nn/BipedAsymm.pth # task.env.learn.reward.dof_pow.scale=-1e-5 

        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_15-22-03-24/nn/BipedAsymm.pth # with additional observations: last_actions

        
        # # task.sim.dt=0.005
        # # task.env.control.decimation=2
        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_18-16-12-11/nn/BipedAsymm.pth 

        # task.sim.dt=0.0025
        # task.env.control.decimation=4
        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_19-18-39-10_dt_0.0025_decimation_4/nn/BipedAsymm.pth

        # task.sim.dt=0.001
        # task.env.control.decimation=10
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_19-15-09-00/nn/BipedAsymm.pth

        # task.sim.dt=0.005
        # task.env.control.decimation=2
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_19-15-09-00/nn/BipedAsymm.pth

        # # # 8/20 adding phase
        # # task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact,heightMap,phase]
        # # task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,phase]
        # checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_20-01-07-41/nn/BipedAsymm.pth # not stable at 0 velocity, tends to rotate


        # # # 8/20 adding contact
        # task.sim.dt=0.0025
        # task.env.control.decimation=4
        # task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact,heightMap]
        # task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact]
        checkpoint=outputs/Biped/train/biped_debug/runs/BipedAsymm_20-11-42-56/nn/BipedAsymm.pth


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

     ++task.env.renderFPS=100
    )
    
    
    BASE_ARGS+=(
        
        # task.env.terrain.maxInitMapLevel=9

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_19-14-02-15/nn/BipedAsymm.pth

        # task.sim.dt=0.0025
        # task.env.control.decimation=4

        # task.sim.dt=0.001
        # task.env.control.decimation=10

        # ++task.env.renderFPS=100

        # task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,last_actions,contactTarget]

        # # 8/20 adding phase
        # task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact,heightMap,phase]
        # task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,phase]

        # # 8/20 adding contact
        task.sim.dt=0.0025
        task.env.control.decimation=4
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact,heightMap]
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact]
    )
}


biped(){
    base
    change_hydra_dir
    PLAY_ARGS+=(
        headless=false
        # pipeline=cpu
        # ++task.env.baseHeightOffset=-0.02
        # task.env.urdfAsset.AssetOptions.fix_base_link=true
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_12-18-53-53/nn/BipedAsymm.pth    # ok, task.env.learn.reward.dof_limit.scale=-100.0 task.env.learn.reward.dof_limit.margin=0.1
        
        # outputs/Biped/train/biped/runs/BipedAsymm_12-22-32-08/config.yaml
        checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_12-22-32-08/nn/BipedAsymm.pth # GOOD!  👍  action filtering: self.actions_filt[:] = self.actions_filt * 0.1 + self.actions * 0.9
        

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_13-12-46-08/nn/BipedAsymm.pth # action filtering: self.actions_filt[:] = self.actions_filt * 0.1 + self.actions * 0.9, and  task.env.randomize.push.interval_s=5
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_15-17-30-58/nn/BipedAsymm.pth # base_height reward, 0.1,100
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_15-18-16-59/nn/BipedAsymm.pth # does not know how to bend knee, base_height reward, 0.1,20
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_15-19-02-15/nn/BipedAsymm.pth #base_height reward, 0.05,20
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_17-11-34-07/nn/BipedAsymm.pth # original check
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_17-11-34-22/nn/BipedAsymm.pth # with groups
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_17-12-22-02/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_17-12-57-34/nn/BipedAsymm.pth # bad step height
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_17-13-42-46/nn/BipedAsymm.pth # non-straight knees when standing, but good walking
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_17-13-35-01_good/nn/BipedAsymm.pth # good but vibrate when stading

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_17-14-07-04/nn/BipedAsymm.pth # original check

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_17-18-57-32/nn/BipedAsymm.pth # x->[-0.2,0.6] # worse

        # outputs/Biped/train/biped/runs/BipedAsymm_18-12-03-04/config.yaml
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_18-12-03-04/nn/BipedAsymm.pth # good! baseline, strong right leg 👍

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_18-17-11-28/nn/BipedAsymm.pth # more push
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_19-13-49-34/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_19-14-02-15/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_19-14-41-24/nn/BipedAsymm.pth

        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_19-15-09-00/nn/BipedAsymm.pth #motor strength

        # # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.2]
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-00-03-19/nn/BipedAsymm.pth # base height changes too much

        # # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.2]
        # # task.env.learn.reward.should_contact.scale=0.5
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-00-12-35/nn/BipedAsymm.pth  # step too low


        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.2]
        # task.env.learn.reward.should_contact.scale=0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-00-21-51/nn/BipedAsymm.pth # step too low
 

        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.2]
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.9
        # task.env.learn.reward.foot_height.scale=0.2
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-00-32-53/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-00-32-53/nn/last_BipedAsymm_ep_700_rew_60.25968.pth

        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.2]
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.9
        # task.env.learn.reward.foot_height.scale=0.2
        # phase_freq: 1.33333333333333 # changed in config.yaml
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-00-45-34/nn/BipedAsymm.pth

        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.2]
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.9
        # task.env.learn.reward.foot_height.scale=0.2
        # phase_freq: 1.33333333333333 # changed in config.yaml
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-00-45-34/nn/BipedAsymm.pth


        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.2]
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5!!
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-01-29-12/nn/BipedAsymm.pth # GOOD IN SIM! STRAIGHT KNEE WALKING. shakes a lot in real



        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0] # remove z vel
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5!!
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-13-55-29/nn/BipedAsymm.pth # result in base moving up/down


        # task.env.learn.reward.action_rate.scale=0.05
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0] # remove z vel
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-15-34-16/nn/BipedAsymm.pth # step height too low.


        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0] # remove z vel
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-15-59-08/nn/BipedAsymm.pth


        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-16-25-18/nn/BipedAsymm.pth

        # task.env.learn.reward.base_height.scale=0.05
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-16-29-47/nn/BipedAsymm.pth # step height too low

        # task.env.learn.reward.base_height.scale=0.01
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-18-34-00/nn/BipedAsymm.pth # very low step height at low velocity command, ok step height at high velocity


        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.base_height.scale=0.01
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-23-47-09/nn/BipedAsymm.pth # very low step height at low velocity command

        # outputs/Biped/train/biped/runs/BipedAsymm_20-23-48-36_OK/config.yaml
        # task.env.learn.reward.foot_height.scale=0.5 # increased
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.base_height.scale=0.01
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_20-23-48-36_OK/nn/BipedAsymm.pth # good step height, but not symmetrical, knee too straight at standing

        # task.env.learn.reward.foot_orientation.exp_scale=-10 #-8
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.base_height.scale=0.01
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-00-04-31/nn/BipedAsymm.pth # step height not as good as previous policy, but more stable

        # task.env.learn.reward.action_rate.scale=0.02
        # task.env.learn.reward.foot_orientation.exp_scale=-10
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.base_height.scale=0.01
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-00-07-25/nn/BipedAsymm.pth # foot too close, not stable


        # outputs/Biped/train/biped/runs/BipedAsymm_21-00-46-18_OK/config.yaml
        # task.env.learn.reward.base_height.scale=0.02
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-00-46-18_OK/nn/BipedAsymm.pth # 👍

        # task.sim.dt=0.001
        # task.env.control.decimation=10
        # ++task.env.renderFPS=100


        # task.env.learn.reward.base_height.exp_scale=-30 
        # task.env.learn.reward.base_height.scale=0.02
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-00-49-01/nn/BipedAsymm.pth # step height too low, it seems the base_height may be too high


        # outputs/Biped/train/biped/runs/BipedAsymm_21-10-18-52/config.yaml
        # task.env.baseHeightTargetOffset=-0.02
        # task.env.learn.reward.base_height.exp_scale=-100 
        # task.env.learn.reward.base_height.scale=0.02
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-10-18-52/nn/BipedAsymm.pth # good except that the standing policy knee it too straight


        # outputs/Biped/train/biped/runs/BipedAsymm_21-10-25-58/config.yaml
        # task.env.learn.reward.base_height.scale=0.05
        # task.env.baseHeightTargetOffset=-0.02
        # task.env.learn.reward.base_height.exp_scale=-100 
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-10-25-58/nn/BipedAsymm.pth # asymetric step height
        
        # outputs/Biped/train/biped/runs/BipedAsymm_21-14-51-49/config.yaml
        # task.env.learn.reward.base_height.exp_scale=-200
        # task.env.learn.reward.base_height.scale=0.05
        # task.env.baseHeightTargetOffset=-0.02
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-14-51-49/nn/BipedAsymm.pth # more asymetric than previous

        # outputs/Biped/train/biped/runs/BipedAsymm_21-15-09-22/config.yaml
        # task.env.learn.reward.base_height.scale=0
        # task.env.learn.reward.base_height.exp_scale=-100
        # task.env.baseHeightTargetOffset=-0.02
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-15-09-22/nn/BipedAsymm.pth # not stepping at all


        # outputs/Biped/train/biped/runs/BipedAsymm_21-17-48-20/config.yaml
        # task.env.learn.reward.dof_limit.scale=-1000
        # task.env.learn.reward.base_height.scale=0
        # task.env.learn.reward.base_height.exp_scale=-100
        # task.env.baseHeightTargetOffset=-0.02
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-17-48-20/nn/BipedAsymm.pth  # knee bend, ok, velocity tracking is bad, so should increase velociy tracking 

        # task.env.learn.reward.dof_limit.scale=-500
        # task.env.learn.reward.base_height.scale=0
        # task.env.learn.reward.base_height.exp_scale=-100
        # task.env.baseHeightTargetOffset=-0.02
        # task.env.learn.reward.foot_height.scale=0.5
        # task.env.learn.reward.foot_forward.exp_scale=-10
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05]
        # task.env.learn.reward.action_rate.scale=0.01 
        # task.env.learn.reward.should_contact.scale=1.0 # clamp min should_contact curriculum 0->0.5
        # task.env.learn.reward.foot_height.scale=0.2
        # task.env.learn.guided_contact.phase_freq=1.33333333333333
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-18-44-11/nn/BipedAsymm.pth # not as good as previous dof_limit.scale=-1000


        # outputs/Biped/train/biped/runs/BipedAsymm_21-23-23-43/config.yaml
        # task.env.learn.reward.dof_limit.scale=-1000
        # task.env.learn.reward.lin_vel.exp_scale=-8  #-4.0
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05] #[1,1,0] #[1,1,0.2] # [1,1,0.1]
        # task.env.learn.reward.ang_vel.exp_scale=-8 #-4.0
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-23-23-43/nn/BipedAsymm.pth # step height is low


        # outputs/Biped/train/biped/runs/BipedAsymm_21-23-31-47/config.yaml
        # task.env.learn.reward.base_height.scale=0.02
        # task.env.learn.reward.dof_limit.scale=-1000
        # task.env.learn.reward.lin_vel.exp_scale=-8  #-4.0
        # task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05] #[1,1,0] #[1,1,0.2] # [1,1,0.1]
        # task.env.learn.reward.ang_vel.exp_scale=-8 #-4.0
        # checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_21-23-31-47/nn/BipedAsymm.pth # knee too straight at low speed


        # outputs/Biped/train/biped/runs/BipedAsymm_20-01-35-49
        # should_contact curriculum 0->0.5 
        # rew["orientation"]=exp_square(torch.arccos(-self.projected_gravity[:, 2]),exp_scale=self.rew_orient_exp_scale) # changed 8/20 
        # NO CHECKPOINT, NOT STABLE

        # outputs/Biped/train/biped/runs/BipedAsymm_22-19-04-45/config.yaml
        # task.env.learn.reward.lin_vel.exp_scale=-4
        # task.env.learn.reward.ang_vel.exp_scale=-4
        # task.env.learn.reward.base_height.exp_scale=-20
        # should_contact curriculum 0->0.3
        checkpoint=outputs/Biped/train/biped/runs/BipedAsymm_22-19-04-45/nn/BipedAsymm.pth # bad very low step height at low speed.

        # checkpoint= # more push 

        # checkpoint=null
        # ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0]
        # task.env.baseHeightOffset=0.1
        # task.env.baseInitState.rot=[0.0998334,0,0,0.9950042] # tilt sideways

        # num_envs=24
        # ++task.env.viewer.follower_offset=[0.3,0.8,0.2]






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

        # # task.sim.dt=0.00125
        # task.sim.dt=0.0025
        # ++task.env.renderFPS=100

        ++task.env.renderFPS=50

        task.env.dataPublisher.enable=true

        task.env.randomize.push.enable=false
        # task.env.randomize.push.enable=true


        task.env.randomize.dof_strength.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false

    )
    BASE_ARGS+=(
        task=Biped
        # headless=true

        ++task.env.max_observation_delay_steps=1 # 1 # 0 observation delay


        task.env.learn.guided_contact.phase_freq=1.33333333333333 # used since 8/20

        # task.env.learn.guided_contact.phase_stance_ratio=0.54 # bad..

        # task.sim.physx.num_position_iterations=4

        train.params.config.horizon_length=32
        train.params.config.mini_epochs=8
        # train.params.config.minibatch_size=16384

        # train.params.config.gamma=0.995

        train.params.config.max_epochs=3000

        # task.env.urdfAsset.file=urdf/v6_vis_URDF_new_body/v6_vis_URDF.urdf

        task.env.randomize.push.enable=true
        task.env.randomize.push.velMin=[-0.2,-0.2,-0.2,-0.4,-0.4,-0.4]
        task.env.randomize.push.velMax=[0.2,0.2,0.2,0.4,0.4,0.4]
        task.env.randomize.push.interval_s=4



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

        # task.env.control.stiffness=60 # used before
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        # ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]

        # Aug 8. dynamics rematched
        task.env.control.stiffness=60
        task.env.control.damping=5
        ++task.env.assetDofProperties.damping=0
        #                                       #0      1      2      3      4      5      6      7      8      9
        ++task.env.assetDofProperties.armature=[0.2100,0.4200,0.3800,0.2100,0.0750,0.2100,0.4200,0.3800,0.2100,0.0750]
        ++task.env.assetDofProperties.friction=[0.0040,0.0100,0.0040,0.0400,0.0200,0.0040,0.0100,0.0040,0.0400,0.0200]
        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]

        # task.env.randomCommandVelocityRanges.linear_x=[-0.2,0.5]
        task.env.randomCommandVelocityRanges.linear_x=[-0.2,0.8] #[-0.2,0.6] #[-0.2,0.8]
        task.env.randomCommandVelocityRanges.linear_y=[-0.2,0.2]
        # task.env.randomCommandVelocityRanges.yaw=[-1,1]
        task.env.randomCommandVelocityRanges.yaw=[-0.8,0.8]

        task.env.randomize.friction.range=[0.2,1.2] # [0,1.0]
        task.env.randomize.baseMass.enable=True
        task.env.randomize.baseMass.range=[2.0,5.0] # [-0.5,2.0] # account for the jetson orin

        task.env.learn.allowKneeContacts=false

        task.env.learn.reward.lin_vel.exp_scale=-4  #-4.0
        task.env.learn.reward.lin_vel.normalize_by=[1,1,0.05] #[1,1,0] #[1,1,0.2] # [1,1,0.1]

        task.env.learn.reward.ang_vel.exp_scale=-4 #-4.0
        
        # task.env.learn.reward.ang_vel.scale=1 #0.5
        
        task.env.learn.reward.orientation.scale=1 #0.5
        task.env.learn.reward.orientation.exp_scale=-10

        task.env.learn.reward.dof_force_target.scale=0.1
        task.env.learn.reward.dof_force_target.exp_scale=-1.0
        task.env.learn.reward.dof_force_target_swing.scale=0 # 0.05 #0
        task.env.learn.reward.dof_force_target_swing.exp_scale=-1.0
        
        task.env.learn.reward.dof_acc.scale=0.1
        task.env.learn.reward.dof_acc.exp_scale=-0.0001   
        task.env.learn.reward.dof_vel.scale=0
        task.env.learn.reward.dof_pos.scale=-0.02 # -0.04 #-0.02 #-0.02 #-0.01
        task.env.learn.reward.dof_pow.scale=0 #-2e-5 #-1e-5 # 0

        task.env.learn.reward.base_height.scale=0.02 #0 # 0.05 #0.02 # 0.01 #0.05 #0
        task.env.learn.reward.base_height.exp_scale=-20 #-100 #-200 #-100 #-30 #-20
        task.env.baseHeightTargetOffset=-0.02



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
        task.env.learn.reward.foot_forward.exp_scale=-10 #-8 #-5 # -10

        task.env.learn.reward.foot_orientation.scale=0.1 # 0.15 #0.1
        task.env.learn.reward.foot_orientation.exp_scale=-8 # -10 #-8 #-5 #-10

        task.env.learn.reward.action_rate.scale=0.01 #0.02 # 0.01 #0.05 # 0 # 0.01 #0.05 #0.1 # 0 # 0.2
        task.env.learn.reward.action_rate.exp_scale=-0.001

        task.env.learn.reward.dof_limit.scale=-1000 #-500 #-100.0
        task.env.learn.reward.dof_limit.margin=0.1

        task.env.learn.reward.should_contact.scale=1.0 #0.5 #1.0 #1.0 # 0.5 result in foot dragging
        task.env.learn.reward.slip.scale=0

        task.env.learn.reward.foot_height.scale=0.5 # 0.2 # 0.1

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


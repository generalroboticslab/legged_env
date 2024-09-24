# #!/bin/bash


dukehumanoid_baseline(){
    base
    change_hydra_dir
    PLAY_ARGS+=(
        headless=false

        checkpoint=assets/checkpoints/dukehumanoid_baseline.pth
        
        num_envs=2
        task.env.dataPublisher.enable=true

        task.env.learn.episodeLength_s=999

        task.env.terrain.maxInitMapLevel=1
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=10
        # task.env.terrain.curriculum=False
        # task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainType=heightfield
        task.env.terrain.terrainType=plane
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,1,0]
        # task.env.terrain.slopeTreshold=0.2
        # task.env.terrain.terrainProportions=[1,1,3,0,0,1,1,3,0]
        # task.env.terrain.terrainProportions=[0,0,0,1,1,0,0,0,0] # stairs only
        # task.env.terrain.terrainProportions=[1,1,0,0,0,0,0,0,0] # rough slop
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,0,0] # rough flat
        # task.env.terrain.terrainProportions=[0,0,0,0,0,1,1,0,0] # smooth slop
        # task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0] # discrete
        task.env.terrain.terrainProportions=[1,1,0,1,1,0,0,1,0] # slop+stairs+discrete

        ++task.env.renderFPS=50

        task.env.randomize.baseInertiaOrigin.enable=false
        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.randomize.dof_strength.enable=false
        task.env.randomize.default_dof_pos.enable=false
        task.env.randomize.link_mass.enable=false
        task.env.randomize.link_inertia.enable=false
        task.env.randomize.body_force.enable=false
        task.env.randomize.baseMass.enable=false
        task.env.learn.addNoise=false
    )

    REWARD_ARGS=(
        task=Biped

        task.env.keep_still_at_zero_command=false
        ++task.env.max_observation_delay_steps=1
        task.env.learn.guided_contact.phase_freq=1.25

        train.params.config.horizon_length=32
        train.params.config.mini_epochs=8
        train.params.config.max_epochs=3000

        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact,phase]
        task.env.observationNames=[angularVelocity,projected_gravity_filtered,commands,dofPosition,dofVelocity,actions,contactTarget,phase]
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[512,256,128]
        ++task.env.num_stacked_obs_frame=5
        ++task.env.num_stacked_state_frame=3

        task.env.randomCommandVelocityRanges.linear_x=[0,1] 
        task.env.randomCommandVelocityRanges.linear_y=[0,0] 
        task.env.randomCommandVelocityRanges.yaw=[0,0]

        task.env.randomize.baseInertiaOrigin.enable=true
        task.env.randomize.baseInertiaOrigin.range=[[-0.02,0.02],[-0.02,0.02],[-0.02,0.02]]
        task.env.randomize.link_inertia.enable=true
        task.env.randomize.link_mass.enable=true

        task.env.randomize.push.enable=true
        task.env.randomize.push.velMin=[-0.2,-0.2,-0.2,-0.4,-0.4,-0.4]
        task.env.randomize.push.velMax=[0.2,0.2,0.2,0.4,0.4,0.4]
        task.env.randomize.push.interval_s=8 

        task.env.randomize.friction.range=[0.2,1.2] # [0,1.0]
        task.env.randomize.baseMass.enable=true
        task.env.randomize.baseMass.range=[-0.5,5.0]
        task.env.randomize.dof_strength.enable=true
        task.env.randomize.dof_strength.range=[0.95,1.02]

        task.env.randomize.action_delay.enable=true

        task.env.learn.allowKneeContacts=false

        task.env.learn.reward.lin_vel.exp_scale=-4.0
        task.env.learn.reward.lin_vel.normalize_by=[1,1,0.1]
        task.env.learn.reward.ang_vel.exp_scale=-8.0
        
        task.env.learn.reward.orientation.scale=-20

        task.env.learn.reward.dof_force_target.scale=0.05
        task.env.learn.reward.dof_force_target.exp_scale=-1
        task.env.learn.reward.dof_force_target_swing.scale=0
        task.env.learn.reward.dof_force_target_swing.exp_scale=-1.0

        # task.env.learn.reward.dof_jerk.scale=0 #0.1
        # task.env.learn.reward.dof_jerk.exp_scale=-1e-6
        task.env.learn.reward.dof_acc.scale=0.1
        task.env.learn.reward.dof_acc.exp_scale=-1e-4

        task.env.learn.reward.dof_vel.scale=0
        task.env.learn.reward.dof_pos.scale=-0.05
        task.env.learn.reward.dof_pow.scale=0

        task.env.learn.reward.base_height.scale=0.1
        task.env.learn.reward.base_height.exp_scale=-2000
        task.env.baseHeightTargetOffset=0 #-0.05

        task.env.learn.reward.air_time.scale=1
        task.env.learn.reward.air_time.offset=-0.3
        task.env.learn.reward.stance_time.scale=0
        task.env.learn.reward.single_contact.scale=0
        task.env.learn.reward.contact_force.scale=0
        task.env.learn.reward.slip.scale=0
        task.env.learn.reward.impact.scale=0
        task.env.learn.reward.stumble.scale=0
        task.env.learn.reward.collision.scale=0
        task.env.learn.reward.action.scale=0

        task.env.learn.reward.foot_pos.scale=0.1
        task.env.learn.reward.foot_pos.exp_scale=-1000 
        task.env.learn.reward.foot_pos.normalize_by=[1,1,0]

        task.env.learn.reward.foot_forward.scale=0.1
        task.env.learn.reward.foot_forward.exp_scale=-10

        task.env.learn.reward.foot_orientation.scale=0.1
        task.env.learn.reward.foot_orientation.exp_scale=-8 

        task.env.learn.reward.foot_height.scale=1.0
        task.env.learn.reward.foot_height.clamp_max=0.05 

        task.env.learn.reward.action_rate.scale=0
        task.env.learn.reward.action_rate.exp_scale=-0.001

        task.env.learn.reward.dof_limit.scale=-100.0 
        task.env.learn.reward.dof_limit.margin=0.1 

        task.env.learn.reward.should_contact.scale=0.5

        task.env.terrain.difficultySale=0.2
        task.env.terrain.curriculum=true
        task.env.terrain.slopeTreshold=0.2
        task.env.terrain.terrainProportions=[1,1,2,1,1,1,1,2,0]
        # task.env.terrain.slopeTreshold=0.05
        # task.env.terrain.terrainType=trimesh
        task.env.terrain.terrainType=plane

        task.env.learn.dofPositionNoise=0.03
        task.env.learn.foot_contact_threshold=20
        task.env.learn.guided_contact.phase_start_with_swing=true


    )

    ROBOT_SPECIFIC_ARGS+=(
        ++task.env.assetDofProperties.velocity=10
        task.env.control.stiffness=[80,80,80,80,60,80,80,80,80,60]
        task.env.control.damping=[8,8,8,8,5,8,8,8,8,5]
        ++task.env.assetDofProperties.damping=0
        #                                       #0      1      2      3      4      5      6      7      8      9
        ++task.env.assetDofProperties.armature=[0.2100,0.4200,0.3800,0.2100,0.0750,0.2100,0.4200,0.3800,0.2100,0.0750]
        # ++task.env.assetDofProperties.friction=[0.0300,0.0300,0.0300,0.0800,0.0800,0.0300,0.0300,0.0300,0.0800,0.0800]
        ++task.env.assetDofProperties.friction=[0.01,0.01,0.01,0.04,0.04,0.01,0.01,0.01,0.04,0.04]

        ++task.env.defaultJointPositions=[0.000,0.175,0.100,0.387,-0.213,0.000,-0.175,-0.100,-0.387,0.213]
    )
    BASE_ARGS+=(${REWARD_ARGS[@]})
    BASE_ARGS+=(${ROBOT_SPECIFIC_ARGS[@]})

    BASE_ARGS+=(

    )
}

dukehumanoid_passive(){
    dukehumanoid_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        # ok
        checkpoint=assets/checkpoints/dukehumanoid_passive.pth
    )

    BASE_ARGS+=(
        ++task.env.learn.action_is_on_min=0 # 0.2
        ++task.env.learn.action_is_on_sigmoid_k=10
        ++task.env.learn.reward.passive_action.scale=0.005
        ++task.env.learn.reward.passive_action_rate.scale=0  #-1e-6
        ++task.env.learn.enablePassiveDynamics=true
        ++task.env.learn.passiveCurriculum=true

    )
}

match_dyn_passive(){
    match_dyn
    BASE_ARGS+=(
        ++task.env.enablePassiveDynamics=true
        ++task.env.learn.reward.passive_action.scale=0.01 # 0.002
    )
}


match_dyn(){
    ENTRY_POINT=dyanmics_matching.py
    change_hydra_dir
    BASE_ARGS+=(
        task=Biped

        test=play
        task.env.terrain.terrainType=plane
        ++task.env.baseHeightOffset=1
        task.env.urdfAsset.AssetOptions.fix_base_link=true
        ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0]
        ++task.env.initialJointPositions=[0,0,0,0,0,0,0,0,0,0]

        task.env.learn.episodeLength_s=555

        task.env.dataPublisher.enable=true

        task.env.randomize.baseInertiaOrigin.enable=false
        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.randomize.dof_strength.enable=false
        task.env.randomize.default_dof_pos.enable=false
        task.env.randomize.link_mass.enable=false
        task.env.randomize.link_inertia.enable=false
        task.env.randomize.body_force.enable=false
        task.env.randomize.baseMass.enable=false
        task.env.randomize.action_delay.enable=false
        task.env.learn.addNoise=false



        task.env.control.actionScale=1
        
        task.env.control.decimation=4
        ++task.env.renderFPS=50

        # task.env.control.decimation=4
        # task.sim.dt=0.0025
        # ++task.env.renderFPS=100


        num_envs=1

        ++task.env.assetDofProperties.velocity=20
        ++task.env.assetDofProperties.stiffness=0
        # ++tast.env.assetDofProperties.effort=200


        # sep 9 adjusted
        # task.env.control.stiffness=0
        # task.env.control.damping=0
        task.env.control.stiffness=[60,60,60,60,60,60,60,60,60,60]
        task.env.control.damping=[5,5,5,5,5,5,5,5,5,5]
        ++task.env.assetDofProperties.damping=0
        #                                       #0      1      2      3      4      5      6      7      8      9
        ++task.env.assetDofProperties.armature=[0.2100,0.4200,0.3800,0.2100,0.0750,0.2100,0.4200,0.3800,0.2100,0.0750]
        ++task.env.assetDofProperties.friction=[0.0300,0.0300,0.0300,0.0800,0.0800,0.0300,0.0300,0.0300,0.0800,0.0800]




    )
}



dog(){
    base
    change_hydra_dir
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
        num_envs=2
        checkpoint=assets/checkpoints/A1Terrain.pth
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
        checkpoint=assets/checkpoints/A1Terrain.pth
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
        checkpoint=assets/checkpoints/AnymalTerrain.pth
       task.env.enableDebugVis=True
        task.env.dataPublisher.enable=true
        num_envs=20
        # task.env.terrain.terrainType=heightfield
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.difficultySale=0.5
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
    EXPORT_ARGS=(
        test=export
        num_envs=1
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

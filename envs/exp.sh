# #!/bin/bash


biped_sim_rew_tune2_guided_contact_asymm_larger_critc_smaller_actor(){ # better than biped_sim_rew_tune2_guided_contact_asymm
    biped_sim_rew_tune2_guided_contact_asymm
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm_larger_critc_smaller_actor/runs/BipedAsymm_17-13-47-18/nn/BipedAsymm.pth

    )
    BASE_ARGS+=(
        train.params.network.mlp.units=[256,128,64]
        train.params.config.central_value_config.network.mlp.units=[1024,512,256]
    )
}

biped_sim_rew_tune2_guided_contact_asymm_larger_critc(){ # better than biped_sim_rew_tune2_guided_contact_asymm
    biped_sim_rew_tune2_guided_contact_asymm
    change_hydra_dir
    PLAY_ARGS+=(
        test=export
        num_envs=1 # exported policy only works with 1 num of env
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm_larger_critc/runs/BipedAsymm_17-13-07-44/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[1024,512,256]
    )
}

biped_sim_rew_tune2_guided_contact_asymm_debug(){
    biped_sim_rew_tune2_guided_contact_terrain
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm_debug/runs/BipedAsymm_18-10-08-58/nn/BipedAsymm.pth
    
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget]
        task.env.stateNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget]
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[512,256,128]
    )
}
# find ./outputs -type d -iname "*biped_sim_rew_tune2_guided_contact_asymm_debug*" -exec find {} -type f -name "config.yaml" \;


biped_sim_rew_tune2_guided_contact_asymm_urdf_20mmfillet_foot(){
    biped_sim_rew_tune2_guided_contact_terrain
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm_urdf_20mmfillet_foot/runs/Biped_19-11-06-25/nn/Biped.pth
    
        # checkpoint=null
        # ++task.env.baseHeightOffset=1
        # task.env.urdfAsset.AssetOptions.fix_base_link=true
        # ++task.env.baseHeightOffset=0.001

    )
    TRAIN_ARGS+=(
        headless=false
    )

    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/v6_vis_URDF_20mmfillet_foot/v6_vis_URDF.urdf
    )
}

biped_sim_rew_tune2_guided_contact_asymm_urdf_foot_thin(){ # good 👍
    biped_sim_rew_tune2_guided_contact_terrain
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm_urdf_foot_thin/runs/Biped_19-13-44-02/nn/Biped.pth

        # test=export
        # num_envs=1 # exported policy only works with 1 num of env

        # task.env.control.stiffness=120.0
        # task.env.control.actionScale=0.0
        # checkpoint=null
        # # ++task.env.baseHeightOffset=1
        # # task.env.urdfAsset.AssetOptions.fix_base_link=true
        # ++task.env.baseHeightOffset=0.001
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/v6_vis_URDF_thin/v6_vis_URDF.urdf
    )
}

biped_sim_rew_tune2_guided_contact_asymm(){
    biped_sim_rew_tune2_guided_contact_terrain
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm/runs/BipedAsymm_11-18-46-57/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm/runs/BipedAsymm_16-17-08-21/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm/runs/BipedAsymm_17-12-44-56/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm/runs/BipedAsymm_17-12-53-58/nn/BipedAsymm.pth
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_asymm/runs/BipedAsymm_19-11-17-31/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact,heightMap]
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[512,256,128]
    )
}

biped_sim_rew_tune2_guided_contact_terrain(){
    biped_sim_rew_tune2_guided_contact
    change_hydra_dir
    PLAY_ARGS+=(
        # task.env.heightmap.x=[0]
        # task.env.heightmap.y=[0]

        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainType=heightfield
        task.env.terrain.terrainType=plane

        
        # test=export
        # num_envs=1 # exported policy only works with 1 num of env

        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_12-19-02-41/nn/Biped.pth

        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_12-19-02-41/nn/last_Biped_ep_550_rew_40.0172.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_12-19-13-48/nn/Biped.pth

        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-13-49-28/nn/Biped.pth

        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-14-52-44/nn/Biped.pth # step height reward ONE FOOT HIGH IN THE AIR..

        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-15-01-06/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-15-33-34/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-16-22-47/nn/Biped.pth # OLD URDF
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_17-14-49-39/nn/Biped.pth # NEW URDF wide feet
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_18-11-05-45/nn/Biped.pth

    )
    BASE_ARGS+=(

        task.env.learn.reward.dof_force_target.scale=0.1

        task.env.terrain.terrainType=trimesh
        task.env.terrain.difficultySale=0.2
        task.env.terrain.curriculum=true
        task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0]
        task.env.terrain.slopeTreshold=0.1


        # task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact]

    )
}

biped_sim_rew_tune2_guided_contact(){ 
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        task.env.enableDebugVis=True

        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-17-14-10/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-17-23-24/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-17-29-06/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-17-40-42/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-17-44-11/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-17-48-51/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-18-06-39/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-18-18-49/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-18-28-09/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-18-42-20/nn/Biped.pth
        ### best
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-18-58-59/nn/Biped.pth # best, no dof_force_target reward
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-19-02-02/nn/Biped.pth # with dof_force_target reward, feet dragging
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-19-11-32/nn/Biped.pth # task.env.learn.reward.dof_pos.scale=-0.05
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-19-14-38/nn/Biped.pth # task.env.learn.reward.dof_pos.scale=-0.02
        
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-19-19-59/nn/Biped.pth # only stand: task.env.learn.reward.dof_pos.scale=-0.1
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_11-19-29-54/nn/Biped.pth #  task.env.learn.reward.dof_pos.scale=-0.2

        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact/runs/Biped_12-18-52-57/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-16-01-09/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-16-06-23/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-16-10-22/nn/Biped.pth

        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_guided_contact_terrain/runs/Biped_16-16-19-21/nn/Biped.pth
    )
    BASE_ARGS+=(
        # # tentatively constrain dof force
        task.env.learn.reward.dof_force_target.scale=0.1

        # task.env.learn.reward.dof_pos.scale=-0.05
        # task.env.learn.reward.dof_pos.scale=-0.02
        # task.env.learn.reward.dof_pos.scale=-0.1
        task.env.learn.reward.dof_pos.scale=-0.2


        task.env.randomize.friction.range=[0,1.0]

        # task.env.learn.reward.should_contact.scale=0.5 # orignal

        task.env.learn.reward.should_contact.scale=1


        task.env.randomCommandVelocityRanges.linear_x=[-0.2,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[-0.2,0.2]
        task.env.randomCommandVelocityRanges.yaw=[-1,1]
        # task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,phase,contactTarget]
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget]

    )
}
biped_sim_rew_tune2_constant_vel_04_guided_contact(){  # walking with single contact
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_constant_vel_04_guided_contact/runs/Biped_11-16-47-19/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2_constant_vel_04_guided_contact/runs/Biped_11-16-36-00/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.should_contact.scale=1
        task.env.randomCommandVelocityRanges.linear_x=[0.4,0.4]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,phase]

    )
}


# baseline 2
biped_sim_rew_tune2(){ # biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10 + no collision
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2/runs/Biped_09-23-16-24/nn/Biped.pth
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune2/runs/Biped_10-00-07-25/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.lin_vel_xy.scale=1
        task.env.learn.reward.lin_vel_xy.expScale=-4
        task.env.learn.allowKneeContacts=false
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
        task.env.learn.reward.orientation.scale=0.5
        task.env.learn.reward.orientation.expScale=-10

        # task.env.learn.reward.lin_vel_xy.scale=1
        # task.env.learn.reward.lin_vel_xy.expScale=-4
        # task.env.learn.reward.lin_vel_z.scale=0
        # task.env.learn.reward.ang_vel_xy.scale=0
        # task.env.learn.reward.ang_vel_z.scale=0
        # task.env.learn.reward.orientation.scale=0
        # task.env.learn.reward.dof_force_target.scale=0
        # task.env.learn.reward.dof_acc.scale=0
        # task.env.learn.reward.dof_vel.scale=0
        # task.env.learn.reward.dof_pos.scale=0
        # task.env.learn.reward.dof_pow.scale=0
        # task.env.learn.reward.base_height.scale=0
        # task.env.learn.reward.air_time.scale=0
        # task.env.learn.reward.stance_time.scale=0
        # task.env.learn.reward.single_contact.scale=0
        # task.env.learn.reward.contact_force.scale=0
        # task.env.learn.reward.impact.scale=0
        # task.env.learn.reward.stumble.scale=0
        # task.env.learn.reward.slip.scale=0
        # task.env.learn.reward.collision.scale=0
        # task.env.learn.reward.action.scale=0
        # task.env.learn.reward.action_rate.scale=0
        # task.env.learn.reward.dof_limit.scale=0
    )
}


biped_sim_rew_tune2_singleContact_01_g00_airtime_s1_o_04(){ # still jumping
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_01_g00_airtime_s1_o_04/runs/Biped_10-02-53-00/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.1
        task.env.learn.reward.single_contact.grace_period=0.0 # no grace
        task.env.learn.reward.air_time.scale=1.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}

biped_sim_rew_tune2_singleContact_01_g02_airtime_s1_o_04(){ # also 2 leg jumping
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_01_g02_airtime_s1_o_04/runs/Biped_10-01-02-42/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.1
        task.env.learn.reward.single_contact.grace_period=0.2
        task.env.learn.reward.air_time.scale=1.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}

biped_sim_rew_tune2_singleContact_05_g02_airtime_s1_o_04(){ # 2 leg jumping
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_05_g02_airtime_s1_o_04/runs/Biped_10-01-01-01/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.5
        task.env.learn.reward.single_contact.grace_period=0.2
        task.env.learn.reward.air_time.scale=1.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}

biped_sim_rew_tune2_singleContact_05_g02_airtime_s5_o_04_more_push(){
    biped_sim_rew_tune2_singleContact_05_g02_airtime_s5_o_04
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_05_g02_airtime_s5_o_04_more_push/runs/Biped_10-10-59-43/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.randomize.push.interval_s=4
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
    )
}

biped_sim_rew_tune2_singleContact_05_g02_airtime_s5_o_04(){ # single leg hopping !!!!
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_05_g02_airtime_s5_o_04/runs/Biped_10-10-35-51/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.5
        task.env.learn.reward.single_contact.grace_period=0.2
        task.env.learn.reward.air_time.scale=5.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}



biped_sim_rew_tune2_singleContact_05_g00(){ # jumping
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_05_g00/runs/Biped_09-23-38-57/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.5
        task.env.learn.reward.single_contact.grace_period=0
    )
}

biped_sim_rew_tune2_singleContact_1_g02(){ # jumping as well
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_1_g02/runs/Biped_10-01-14-59/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=1.0
        task.env.learn.reward.single_contact.grace_period=0.2
    )
}

biped_sim_rew_tune2_singleContact_05_g02_strict(){ # stepping, small steps
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_05_g02_strict/runs/Biped_10-10-22-04/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.5
        task.env.learn.reward.single_contact.grace_period=0.2
    )
}

biped_sim_rew_tune2_singleContact_05_g02(){ # stepping, small steps
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_singleContact_05_g02/runs/Biped_09-23-37-34/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.5
        task.env.learn.reward.single_contact.grace_period=0.2
    )
}

biped_sim_rew_tune2_airtime_s10_o_04(){ # jumping, air time will cause wider stance
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_airtime_s10_o_04/runs/Biped_09-23-31-18/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.air_time.scale=10.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}

biped_sim_rew_tune2_airtime_s5_o_04(){  # jumping, air time will cause wider stance
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_airtime_s5_o_04/runs/Biped_09-23-31-11/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.air_time.scale=5.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}


biped_sim_rew_tune2_dofForceTarget_01(){  # jumping, with old torque reward
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_dofForceTarget_01/runs/Biped_10-02-22-48/nn/Biped.pth
    )
    BASE_ARGS+=(
        # task.env.learn.reward.dof_force_target.scale=-1e-5
        task.env.learn.reward.dof_force_target.scale=0.1
    )
}
biped_sim_rew_tune2_dofForceTarget_00(){  # jumping, with old torque reward
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_dofForceTarget_01/runs/Biped_10-02-22-48/nn/Biped.pth
    )
    BASE_ARGS+=(
        # task.env.learn.reward.dof_force_target.scale=-1e-5
        task.env.learn.reward.dof_force_target.scale=0.0
    )
}

biped_sim_rew_tune2_airtime_torque_1(){  # jumping, with old torque reward
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(

        # ++task.env.baseHeightOffset=1
        # task.env.urdfAsset.AssetOptions.fix_base_link=true

        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_airtime_torque_1/runs/Biped_10-01-28-56/nn/Biped.pth
    )
    BASE_ARGS+=(
        # task.env.learn.reward.dof_force_target.scale=-1e-5
        task.env.learn.reward.dof_force_target.scale=0.1
    )
}

biped_sim_rew_tune2_constant_vel_04(){  # jumping, with old torque reward
    biped_sim_rew_tune2
    change_hydra_dir
    PLAY_ARGS+=(

        # ++task.env.baseHeightOffset=1
        # task.env.urdfAsset.AssetOptions.fix_base_link=true

        checkpoint=outputs/Biped/train/biped_sim_rew_tune2_constant_vel_04/runs/Biped_10-02-58-08/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.randomCommandVelocityRanges.linear_x=[0.4,0.4]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]
    )
}

biped_sim_backward(){
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_baseline/runs/Biped_09-11-15-02/nn/Biped.pth
    )
    TRAIN_ARGS+=(
        headless=false
    )
    BASE_ARGS+=(

    )
}


biped_sim_asym_more_push(){ # more push result in small steps
    biped_sim_asymm
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_asym_more_push/runs/BipedAsymm_09-11-54-40/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.randomize.push.interval_s=5
        task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]
    )

}

biped_sim_single_contact_air_time_x4_changed_zero_cmd(){ # constant 1 reward for zero command, this is too noisy.
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_single_contact_air_time_x4_changed_zero_cmd/runs/Biped_09-14-22-57/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=1.0
        task.env.learn.reward.single_contact.grace_period=0
        task.env.learn.reward.air_time.scale=4.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}

biped_sim_single_contact_air_time_x4(){ # leg wide apart
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_single_contact_air_time_x4/runs/Biped_09-13-08-16/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=1.0
        task.env.learn.reward.single_contact.grace_period=0
        task.env.learn.reward.air_time.scale=4.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}

biped_sim_single_contact_air_time_x2(){ # still walks on toes
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_single_contact_air_time_x2/runs/Biped_09-13-07-20/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=1.0
        task.env.learn.reward.single_contact.grace_period=0
        task.env.learn.reward.air_time.scale=2.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}

biped_sim_single_contact_air_time(){ # still walks on toes
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_single_contact_air_time/runs/Biped_09-13-05-59/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=1.0
        task.env.learn.reward.single_contact.grace_period=0
        task.env.learn.reward.air_time.scale=1.0
        task.env.learn.reward.air_time.offset=-0.4
    )
}


biped_sim_single_contact(){ # walks on toes
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_single_contact/runs/Biped_09-13-03-12/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=1.0
        task.env.learn.reward.single_contact.grace_period=0
    )
}


biped_sim_asym_single_contact(){ # more push result in small steps
    biped_sim_asymm
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_asym_single_contact/runs/BipedAsymm_09-12-37-29/nn/BipedAsymm.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=1.0
        task.env.learn.reward.single_contact.grace_period=0
    )

}


biped_sim_asymm(){
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_sim_asymm/runs/BipedAsymm_09-11-39-44/nn/BipedAsymm.pth
        # checkpoint=outputs/Biped/train/biped_sim_asymm/runs/BipedAsymm_09-11-39-44/nn/last_BipedAsymm_ep_1500_rew_24.801708.pth
        checkpoint=outputs/Biped/train/biped_sim_asymm/runs/BipedAsymm_09-11-39-44/nn/last_BipedAsymm_ep_1000_rew_23.981369.pth
    )
    BASE_ARGS+=(
        train=BipedPPOAsymm
        task.env.asymmetric_observations=True
        task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        train.params.network.mlp.units=[512,256,128]
        train.params.config.central_value_config.network.mlp.units=[512,256,128]
    )
}


biped_sim_vis(){
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_vis/runs/Biped_09-15-22-25/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/v6_vis_URDF_IMU_top/v6_vis_URDF.urdf
    )
}


biped_sim_vis_IMU_top(){
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=
    )
    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/v6_vis_URDF_IMU_top/v6_vis_URDF.urdf
    )
}


biped_sim_rew_tune_linearVelocityXY(){
    biped_sim_rew_tune_baseline
    change_hydra_dir
    BASE_ARGS+=(
        task.env.learn.reward.lin_vel_xy.scale=1
        task.env.learn.reward.lin_vel_xy.expScale=-5
    )
}


biped_sim_rew_tune_linearVelocityXY_feetSingleContact_grace0(){
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_linearVelocityXY_feetSingleContact_grace0/runs/Biped_09-17-18-38/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.lin_vel_xy.scale=1
        task.env.learn.reward.lin_vel_xy.expScale=-5
        task.env.learn.reward.single_contact.scale=0.5
        task.env.learn.reward.single_contact.grace_period=0

    )
}
biped_sim_rew_tune_linearVelocityXY_feetSingleContact(){ # jumping or small steps
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_linearVelocityXY_feetSingleContact/runs/Biped_09-17-04-26/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.lin_vel_xy.scale=1
        task.env.learn.reward.lin_vel_xy.expScale=-5
        task.env.learn.reward.single_contact.scale=0.5

    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_30_no_collision(){ # legs collision
    biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_30
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_30_no_collision/runs/Biped_09-21-04-11/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.allowKneeContacts=false
    )
}


biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_30(){ # legs collision
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_30/runs/Biped_09-19-30-21/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
        task.env.learn.reward.orientation.scale=0.5
        task.env.learn.reward.orientation.expScale=-30
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_20(){ # also jumping behavior, leg distance ok
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_20/runs/Biped_09-19-53-56/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
        task.env.learn.reward.orientation.scale=0.5
        task.env.learn.reward.orientation.expScale=-20
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_1(){ # another jumping behavior, leg too close
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_1/runs/Biped_09-19-46-54/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
        task.env.learn.reward.orientation.scale=1.0
        task.env.learn.reward.orientation.expScale=-10
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_no_collision(){ # 
    biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_no_collision/runs/Biped_09-20-42-42/nn/Biped.pth
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_no_collision/runs/Biped_09-20-42-55/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.allowKneeContacts=false
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_seed_1(){ # similar to seed 42
    biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_seed_1/runs/Biped_09-21-45-49/nn/Biped.pth
    )
    BASE_ARGS+=(
        seed=1
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_seed_0_no_collision(){
    biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_seed_0
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_seed_0_no_collision/runs/Biped_09-22-06-38/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.allowKneeContacts=false
    )
}


biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_seed_0(){ # bad posture
    biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10_seed_0/runs/Biped_09-21-45-44/nn/Biped.pth
    )
    BASE_ARGS+=(
        seed=0
    )
}

# TRY THIS NEXT
biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10(){ # also small quick steps, good, try this next?
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_exp_orient_10/runs/Biped_09-19-30-13/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
        task.env.learn.reward.orientation.scale=0.5
        task.env.learn.reward.orientation.expScale=-10
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_orient_1_0_no_collision(){ # jumping, allowKneeContacts=false seems to help in symmetry
    biped_sim_rew_tune_actionRate_angularVelocityZ_orient_1_0
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_orient_1_0_no_collision/runs/Biped_09-21-23-53/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.allowKneeContacts=false
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_orient_1_0(){ # no jump, small quick steps TODO COMPARE with biped_sim_rew_tune_actionRate_angularVelocityZ
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_orient_1_0/runs/Biped_09-19-22-24/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
        task.env.learn.reward.orientation.scale=-1.0
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_orient_0_5_no_collision(){ # small steps, no jumping
    biped_sim_rew_tune_actionRate_angularVelocityZ_orient_0_5
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_orient_0_5_no_collision/runs/Biped_09-21-23-23/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.allowKneeContacts=false
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ_orient_0_5(){ # funny jumping behavior resulted from just just 4 rewards (linear velocity xy, angular veloxity z, action rate, and orientation)
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_orient_0_5/runs/Biped_09-19-21-55/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
        task.env.learn.reward.orientation.scale=-0.5
    )
}


biped_sim_rew_tune_actionRate_angularVelocityZ_smalldefaultJointPositions(){ # leg spreading, bad
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ_smalldefaultJointPositions/runs/Biped_09-19-09-12/nn/Biped.pth
        # pipeline="cpu"

    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
        ++task.env.defaultJointPositions=[0.000,0.087,0.000,0.197,-0.110,0.000,-0.087,0.000,-0.197,0.110]
    )
}

biped_sim_rew_tune_actionRate_angularVelocityZ(){ # adding angular velocity z will regularize the action, i.e. reduce bounds loss
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_angularVelocityZ/runs/Biped_09-17-40-21/nn/Biped.pth
        # pipeline="cpu"

    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        task.env.learn.reward.ang_vel_z.scale=0.5
    )
}


biped_sim_rew_tune_actionRate_small(){ # worse pose than action_rate.scale=-1.0e-4
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_small/runs/Biped_09-17-50-53/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-5.0e-5
    )
}



biped_sim_rew_tune_actionRate_smaller_defaultJointPositions(){ # too small, bad pose
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_smaller_defaultJointPositions/runs/Biped_09-18-34-26/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        # even smaller default joint position
        ++task.env.defaultJointPositions=[0.0000,0.0436,0.0000,0.0897,-0.0460,0.0000,-0.0436,0.0000,-0.0897,0.0460]
        # smaller default joint positions
        # ++task.env.defaultJointPositions=[0.000,0.087,0.000,0.197,-0.110,0.000,-0.087,0.000,-0.197,0.110]
        # original default joint positions
        # ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]
    )
}

biped_sim_rew_tune_actionRate_small_defaultJointPositions(){
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate_small_defaultJointPositions/runs/Biped_09-18-13-17/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
        # smaller default joint positions
        ++task.env.defaultJointPositions=[0.000,0.087,0.000,0.197,-0.110,0.000,-0.087,0.000,-0.197,0.110]
        # original default joint positions
        # ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]
    )
}

biped_sim_rew_tune_actionRate(){ # adding action rate will make it learn slower
    biped_sim_rew_tune_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_actionRate/runs/Biped_09-17-29-15/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.action_rate.scale=-1.0e-4
    )
}

biped_sim_rew_tune_baseline(){
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_rew_tune_baseline/runs/Biped_09-16-20-09/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.lin_vel_xy.scale=1
        task.env.learn.reward.lin_vel_xy.expScale=-4
        task.env.learn.reward.lin_vel_z.scale=0
        task.env.learn.reward.ang_vel_xy.scale=0
        task.env.learn.reward.ang_vel_z.scale=0
        task.env.learn.reward.orientation.scale=0
        task.env.learn.reward.dof_force_target.scale=0
        task.env.learn.reward.dof_acc.scale=0
        task.env.learn.reward.dof_vel.scale=0
        task.env.learn.reward.dof_pos.scale=0
        task.env.learn.reward.dof_pow.scale=0
        task.env.learn.reward.base_height.scale=0
        task.env.learn.reward.air_time.scale=0
        task.env.learn.reward.stance_time.scale=0
        task.env.learn.reward.single_contact.scale=0
        task.env.learn.reward.contact_force.scale=0
        task.env.learn.reward.impact.scale=0
        task.env.learn.reward.stumble.scale=0
        task.env.learn.reward.slip.scale=0
        task.env.learn.reward.collision.scale=0
        task.env.learn.reward.action.scale=0
        task.env.learn.reward.action_rate.scale=0
        task.env.learn.reward.dof_limit.scale=0
    )
}

biped_sim_slip(){
    biped_sim_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_slip/runs/Biped_09-16-10-53/nn/Biped.pth
    )
    BASE_ARGS+=(
        task.env.learn.reward.slip.scale=-0.2
    )

}

biped_sim_baseline(){
    base
    task=Biped
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_sim_baseline/runs/Biped_09-11-15-02/nn/Biped.pth
        headless=false

        num_envs=5
        task.sim.dt=0.0025
        ++task.env.renderFPS=100
        # ++task.env.renderFPS=50
        task.env.dataPublisher.enable=true

        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false
    )
    TRAIN_ARGS+=(
        # headless=false
    )
    BASE_ARGS+=(

        headless=true

        task.env.terrain.terrainType=plane
        ++task.env.assetDofProperties.velocity=10
        ++task.env.assetDofProperties.damping=[0.8500,0.4200,0.6000,0.8500,0.9000,0.8500,0.4200,0.6000,0.8500,0.9000]
        ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        ++task.env.assetDofProperties.friction=[0.0115,0.0050,0.0150,0.1300,0.3500,0.0115,0.0050,0.0150,0.1300,0.3500]

        train.params.config.max_epochs=1500

        task.sim.physx.num_position_iterations=2

        task.env.urdfAsset.file=urdf/v6_vis_URDF_wide_feet/v6_vis_URDF.urdf
        # task.env.urdfAsset.file=urdf/biped_long_foot/biped_v6_terrain.urdf
        ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_convex_hulls=1
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_num_vertices_per_ch=64

        # ++task.env.baseHeightOffset=0.1
        ++task.env.baseHeightOffset=0.05
        ++task.env.defaultJointPositions=[0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213]

        task.env.randomCommandVelocityRanges.linear_x=[-0.1,0.5]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]

        task.env.terrain.terrainType=plane
        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.difficultySale=0.2
        # task.env.terrain.curriculum=true
        # task.env.terrain.terrainProportions=[0,0,1,0,0,0,0,0,0]

        task.env.control.stiffness=60.0
        task.env.control.actionScale=1.0

        # train=BipedPPOAsymm
        # task.env.asymmetric_observations=True
        # task.env.observationNames=[angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        # task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions]
        # train.params.network.mlp.units=[512,256,128]
        # train.params.config.central_value_config.network.mlp.units=[1024,512,256]
        
        task.env.learn.reward.slip.scale=-0.1

        # task.env.randomize.push.interval_s=5
        # task.env.randomize.push.velMin=[-0.3,-0.3,0,-0.3,-0.3,-0.3]
        # task.env.randomize.push.velMax=[0.3,0.3,0,0.3,0.3,0.3]

        task.env.randomize.erfi.enable=true
        task.env.randomize.erfi.rfi_range=[-3.0,3.0]
        task.env.randomize.erfi.rao_range=[-2.0,2.0]

        # task.env.learn.dofPositionNoise=0.02
    )
}

biped_urdf_check(){
    biped_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_urdf_check/runs/BipedAsymm_08-10-48-41/nn/last_BipedAsymm_ep_5000_rew_24.6043.pth
    )
    TRAIN_ARGS+=(
        headless=false
    )

    BASE_ARGS+=(
        task.env.urdfAsset.file=urdf/biped_long_foot_6_2/biped_v6_terrain.urdf
    )
}



biped_stepping_stone_slippery_low_torque_reward(){
    biped_stepping_stone_trimesh_no_push
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_stepping_stone_slippery_low_torque_reward/runs/BipedAsymm_08-18-44-44/nn/BipedAsymm.pth
    )
    TRAIN_ARGS+=(
        headless=true
    )

    BASE_ARGS+=(
        task.env.randomize.friction.range=[0,1.0]
        task.env.learn.reward.dof_force_target.scale=-0.00001
    )
}

biped_stepping_stone_slippery_single_contact_0_1(){
    biped_stepping_stone_slippery
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_stepping_stone_slippery_single_contact_0_1/runs/BipedAsymm_09-01-25-19/nn/BipedAsymm.pth
    )
    TRAIN_ARGS+=(
        headless=true
    )

    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.1
    )
}

biped_stepping_stone_slippery_single_contact_0_5(){
    biped_stepping_stone_slippery
    change_hydra_dir
    PLAY_ARGS+=(
        checkpoint=outputs/Biped/train/biped_stepping_stone_slippery_single_contact_0_5/runs/BipedAsymm_09-01-25-38/nn/last_BipedAsymm_ep_5000_rew_32.127003.pth
    )
    TRAIN_ARGS+=(
        headless=true
    )

    BASE_ARGS+=(
        task.env.learn.reward.single_contact.scale=0.5
    )
}

biped_stepping_stone_slippery(){
    biped_stepping_stone_trimesh_no_push
    change_hydra_dir
    PLAY_ARGS+=(
    )
    TRAIN_ARGS+=( 
    )

    BASE_ARGS+=(
        task.env.randomize.friction.range=[0,1.0]
    )
}



biped_stepping_stone_trimesh_no_push(){
    biped_baseline
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_stairs_trimesh_no_push/runs/BipedAsymm_08-13-09-11/nn/last_BipedAsymm_ep_3500_rew_23.80447.pth
        checkpoint=outputs/Biped/train/biped_stepping_stone_trimesh_no_push/runs/BipedAsymm_08-16-50-10/nn/last_BipedAsymm_ep_4700_rew_24.054665.pth
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.terrainType=trimesh
        task.env.terrain.terrainType=plane

        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false


    )
    TRAIN_ARGS+=(
        headless=false
    )

    BASE_ARGS+=(        
        task.env.randomize.push.enable=false

        task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0]
        task.env.terrain.terrainType=trimesh
        task.env.terrain.slopeTreshold=0.1
    )
}



biped_stairs_trimesh_no_push(){
    biped_stairs_trimesh
    change_hydra_dir
    PLAY_ARGS+=(
        # checkpoint=outputs/Biped/train/biped_stairs_trimesh_no_push/runs/BipedAsymm_08-13-09-11/nn/last_BipedAsymm_ep_850_rew_22.430407.pth # feet apart, bad
        # checkpoint=outputs/Biped/train/biped_stairs_trimesh_no_push/runs/BipedAsymm_08-13-09-11/nn/last_BipedAsymm_ep_1250_rew_23.542065.pth
        checkpoint=outputs/Biped/train/biped_stairs_trimesh_no_push/runs/BipedAsymm_08-13-09-11/nn/last_BipedAsymm_ep_3500_rew_23.80447.pth
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        # task.env.terrain.terrainType=trimesh
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
        # checkpoint=outputs/Biped/train/biped_baseline/runs/BipedAsymm_07-20-21-40/nn/BipedAsymm.pth # terrain horizontal_scale=0.1
        checkpoint=outputs/Biped/train/biped_baseline/runs/BipedAsymm_07-20-21-40/nn/last_BipedAsymm_ep_1500_rew_23.810951.pth
        task.env.terrain.terrainType=plane

        # task.env.terrain.terrainType=heightfield
        # task.env.terrain.numLevels=2
        # task.env.terrain.numTerrains=5
        # task.env.terrain.curriculum=False
        # # task.env.terrain.horizontalScale=0.05

        # task.env.terrain.terrainType=trimesh
        # task.env.terrain.terrainProportions=[0,0,0,0,0,0,0,1,0]

        # # terrain types: [rough_up, rough_down, rough_flat, stair_up, stair_down, smooth_up, smooth_down, discrete, stepping_stone]
        # task.env.terrain.terrainProportions=[0,0,0,1,0,0,0,0,0]

        task.env.terrain.difficultySale=0.2

        task.env.dataPublisher.enable=true

        task.sim.dt=0.0025
        # ++task.env.renderFPS=100
        # ++task.env.renderFPS=50
        
        num_envs=20
        headless=false
    )

    BASE_ARGS+=(

        headless=true
        train.params.config.max_epochs=5000

        task.sim.physx.num_position_iterations=2

        task.env.urdfAsset.file=urdf/v6_vis_URDF_wide_feet/v6_vis_URDF.urdf

        # task.env.urdfAsset.file=urdf/biped_long_foot/biped_v6_terrain.urdf
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

        task.env.learn.reward.slip.scale=-0.1
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


biped_c2r5(){ # can walk in small steps. use this as baseline 👍, asymmetric obs, slip=-0.1
    biped
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
        task.env.learn.reward.slip.scale=-0.1

        task.env.control.stiffness=60.0
        task.env.control.actionScale=1.0

        # # for logging names
        # "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[0]}"
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
        ++task.env.baseHeightOffset=0.001

        # offset=0
        # ++task.env.defaultJointPositions=[0.0000,0.0436,0.0000,0.0956,-0.0520,0.0000,-0.0436,0.0000,-0.0956,0.0520]
        # offset=0.3
        # ++task.env.defaultJointPositions=[0.0000,0.0436,0.0000,0.4323,-0.3886,0.0000,-0.0436,0.0000,-0.4323,0.3886]
        # offset=-0.005
        ++task.env.defaultJointPositions=[0.0000,0.0436,0.0000,0.0897,-0.0460,0.0000,-0.0436,0.0000,-0.0897,0.0460]







        task.env.control.stiffness=60.0
        task.env.control.actionScale=1.0

        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.learn.addNoise=false
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
        
        checkpoint=outputs/RobotDog/train/20240710_012200/runs/RobotDog_10-01-22-00/nn/RobotDog.pth
        num_envs=2
        task.env.dataPublisher.enable=true
        task.env.learn.episodeLength_s=999
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
        checkpoint=outputs/A1Terrain/train/20240706_220324/runs/A1Terrain_06-22-03-24/nn/A1Terrain.pth
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
        # checkpoint=outputs/A1Terrain/train/20240706_220324/runs/A1Terrain_06-22-03-24/nn/A1Terrain.pth
        # task.env.dataPublisher.enable=true
        # ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        task.env.enableDebugVis=True
        task.env.dataPublisher.enable=true
        num_envs=20
        task.env.terrain.terrainType=heightfield
        task.env.terrain.numLevels=2
        task.env.terrain.numTerrains=5
        task.env.terrain.curriculum=False
        task.env.terrain.difficultySale=0.5

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


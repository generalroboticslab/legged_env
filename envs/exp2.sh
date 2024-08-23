# #!/bin/bash





# base(){
#     checkpoint=null
#     ENTRY_POINT=train.py
#     PLAY_ARGS=(
#         test=play
#     )
    
#     TRAIN_ARGS=(
#         headless=true
#     )
#     BASE_ARGS=()
#     KEYBOARD_ARGS=(
#         task.env.viewer.keyboardOperator=true
#     )
# }


# change_hydra_dir(){
#     BASE_ARGS+=(
#         "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[1]}"
#     )
# }

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

        # task.env.urdfAsset.AssetOptions.override_inertia=true # HACK

        task.env.terrain.terrainType=plane
        ++task.env.baseHeightOffset=1
        task.env.urdfAsset.AssetOptions.fix_base_link=true
        ++task.env.defaultJointPositions=[0,0,0,0,0,0,0,0,0,0]
        ++task.env.initialJointPositions=[0,0,0,0,0,0,0,0,0,0]

        task.env.learn.episodeLength_s=555

        task.env.urdfAsset.file=urdf/v6_vis_URDF_new_body/v6_URDF_aug10_copy.urdf

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

        # task.env.control.decimation=4
        # task.sim.dt=0.0025
        # ++task.env.renderFPS=100


        num_envs=1

        ++task.env.assetDofProperties.velocity=20
        ++task.env.assetDofProperties.stiffness=0
        # ++tast.env.assetDofProperties.effort=200


 # ---------------------------------------------------------------------------------------------
        # Aug 8. dynamics rematched
        task.env.control.stiffness=60
        task.env.control.damping=5
        ++task.env.assetDofProperties.damping=0
        #                                       #0      1      2      3      4      5      6      7      8      9
        ++task.env.assetDofProperties.armature=[0.2100,0.4200,0.3800,0.2100,0.0750,0.2100,0.4200,0.3800,0.2100,0.0750]
        ++task.env.assetDofProperties.friction=[0.0040,0.0100,0.0040,0.0400,0.0200,0.0040,0.0100,0.0040,0.0400,0.0200]

 #----------------------------------------------------------------------------------------------
        ## Aug 6. dynamics rematched

        # # Note: only for motor 1 and 6
        # task.env.control.stiffness=60
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        # ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]


        # task.env.control.stiffness=60
        # task.env.control.damping=6
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        # ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]

        # task.env.control.stiffness=60
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        # ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]

        # task.env.control.stiffness=60
        # task.env.control.damping=4
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
        # ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]

        # task.env.control.stiffness=50
        # task.env.control.damping=5
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3200,0.3800,0.3200,0.3200,0.0900,0.3200,0.3800,0.3200,0.3200,0.0900]
        # ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]

        # task.env.control.stiffness=50
        # task.env.control.damping=4
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.3600,0.4500,0.3600,0.3600,0.1200,0.3600,0.4500,0.3600,0.3600,0.1200]
        # ++task.env.assetDofProperties.friction=[0.0150,0.0400,0.0150,0.0300,0.1500,0.0150,0.0400,0.0150,0.0300,0.1500]

        
        #----------------------------------------------------------------------------------------------


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

        # task.env.control.stiffness=140
        # task.env.control.damping=7
        # ++task.env.assetDofProperties.damping=0
        # ++task.env.assetDofProperties.armature=[0.2880,0.3600,0.2880,0.2880,0.0960,0.2880,0.3600,0.2880,0.2880,0.0960]
        # ++task.env.assetDofProperties.friction=[0.0115,0.0200,0.0115,0.0300,0.1500,0.0115,0.0200,0.0115,0.0300,0.1500]



        
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


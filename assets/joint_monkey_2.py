"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
"""

import os
# os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib"
# print(os.environ['LD_LIBRARY_PATH'])

import numpy as np
from isaacgym import gymapi, gymutil

# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

asset_descriptors = [   
    AssetDesc("urdf/biped/biped_v6.urdf", False),
    AssetDesc("urdf/RobotDog/RobotDog3kg.urdf", False),
    AssetDesc("urdf/a1/a1_minimum.urdf", False),
    AssetDesc("urdf/a1/a1_minimum_anymal_like.urdf", False),
    AssetDesc("urdf/anymal_c/urdf/anymal.urdf", False),
    AssetDesc("urdf/anymal_c/urdf/anymal_minimal.urdf", False),
    AssetDesc("urdf/anymal_c/urdf/anymal_minimal_a1_like.urdf", False),
    AssetDesc("assets/urdf/biped_long_foot/biped_v6_long.urdf", True)
]

asset_id = 0
compute_device_id = 0
graphics_device_id = 0
physics_engine = gymapi.SIM_PHYSX

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 10e6
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0,0,-10)

sim_params.use_gpu_pipeline = False  #  Forcing CPU pipeline
sim_params.substeps = 1
sim_params.physx.solver_type = 0
sim_params.physx.num_position_iterations = 1
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.use_gpu = False
sim_params.physx.contact_offset=0.02
sim_params.physx.num_threads=1
sim_params.physx.rest_offset=0


# set up the env grid
num_envs = 1
num_per_row = 2
spacing = 1.
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)


# load asset
asset_root = os.path.abspath(os.path.dirname(__file__))
asset_file = asset_descriptors[0].file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[0].flip_visual_attachments
asset_options.use_mesh_materials = True
asset_options.collapse_fixed_joints = True
asset_options.replace_cylinder_with_capsule = False
asset_options.override_inertia = False
asset_options.vhacd_enabled = False
asset_options.disable_gravity = True
asset_options.enable_gyroscopic_forces=False

asset_options.density = 0.001
asset_options.angular_damping = 100.0
asset_options.linear_damping = 100.0
asset_options.armature = 0.0 # DOF armature, a value added to the diagonal of the joint-space inertia matrix. Physically, it corresponds to the rotating part of a motor - which increases the inertia of the joint, even when the rigid bodies connected by the joint can have very little inertia.
asset_options.thickness = 0.01

for k in range(50):
    # initialize gym
    gym = gymapi.acquire_gym()

    # sim_params.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP
    #CC_NEVER : Don’t collect any contacts (value = 0).
    # CC_LAST_SUBSTEP : Collect contacts for last substep only (value = 1).
    # CC_ALL_SUBSTEPS : Collect contacts for all substeps (value = 2) (default).
    
    sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    # # add ground plane
    # plane_params = gymapi.PlaneParams()
    # plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    # gym.add_ground(sim, plane_params)

    # create viewer
    camera_properties = gymapi.CameraProperties()
    # camera_properties.supersampling_horizontal = 2
    # camera_properties.supersampling_vertical = 2

    viewer = gym.create_viewer(sim, camera_properties)
    if viewer is None:
        print("*** Failed to create viewer")
        quit()


    # position the camera
    cam_pos = gymapi.Vec3(0, -0.5, 1.3)
    cam_target = gymapi.Vec3(0, 0, 1.15)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # cache useful handles
    envs = []
    actor_handles = []

    #------------------------------------------------------------
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # get array of DOF names
    dof_names = gym.get_asset_dof_names(asset)

    # get array of DOF properties
    dof_props = gym.get_asset_dof_properties(asset)

    # Maps rigid body names to asset-relative indices
    rigid_body_dict = gym.get_asset_rigid_body_dict(asset)

    # Maps joint names to asset-relative indices
    joint_dict = gym.get_asset_joint_dict(asset)

    # create an array of DOF states that will be used to update the actors
    num_dofs = gym.get_asset_dof_count(asset)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

    # get list of DOF types
    dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

    # get the position slice of the DOF state array
    dof_positions = dof_states['pos']

    # get the limit-related slices of the DOF properties array
    has_limits = dof_props['hasLimits']
 
    # dof_props['stiffness'][:]=100 # HACK
    dof_props['damping'][:] = 100
    dof_props['friction'][:] = 100

    # Print DOF properties
    for i in range(num_dofs):
        print("DOF %d" % i)
        print("  Name:     '%s'" % dof_names[i])
        print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
        print("  Stiffness:  %r" % dof_props['stiffness'][i])
        print("  Damping:  %r" % dof_props['damping'][i])
        print("  Armature:  %r" % dof_props['armature'][i])
        print("  Limited?  %r" % has_limits[i])
        if has_limits[i]:
            print("    Lower   %f" % dof_props['lower'][i])
            print("    Upper   %f" % dof_props['upper'][i])

    # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.32)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1) #xyzw

        actor_handle = gym.create_actor(env, asset, pose, "actor", i, 0)
        actor_handles.append(actor_handle)

        gym.set_actor_dof_properties(env,actor_handle,dof_props)

        # set default DOF positions
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    #-------------------
    rigid_body_names =  gym.get_actor_rigid_body_names(env,actor_handle)
    rigid_body_properties = gym.get_actor_rigid_body_properties(env,actor_handle)
    rigid_body_masses = [p.mass for p in rigid_body_properties]
    rigid_body_inertias = [p.inertia for p in rigid_body_properties]

    total_mass = np.sum(rigid_body_masses)
    for name,mass in zip(rigid_body_names,rigid_body_masses):
        print(f"{name:20s},{mass:.5f}")
    print(f"total mass: {total_mass} kg")

    print(f"{'name':20s}:{'i.x.x':<11s},{'i.x.y':<11s},{'i.x.z':<11s},{'i.y.y':<11s},{'i.y.z':<11s},{'i.z.z':<11s}")
    for name,i in zip(rigid_body_names,rigid_body_inertias):
        print(f"{name:20s}:{i.x.x:<+8.4e},{i.x.y:<+8.4e},{i.x.z:<+8.4e},{i.y.y:<+8.4e},{i.y.z:<+8.4e},{i.z.z:<+8.4e}")

    # -----------------------------------------------------------------------------

    # subscribe to keyboard events
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

    init = False
    should_close = False

    while not should_close:
        should_close = gym.query_viewer_has_closed(viewer)
        # Get input actions from the viewer and handle them appropriately
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "reset" and evt.value > 0:
                print("reset pressed")
                init = True
        if init:
            break


        # dof_states['pos'][:] = np.array([0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213])
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)



        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.clear_lines(viewer)
        # clone actor state in all of the environments
        for i in range(num_envs):
            # gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)
            #show all axis:
            for k in range(num_dofs):
                # get the DOF frame (origin and axis)
                dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], k)
                frame = gym.get_dof_frame(envs[i], dof_handle)

                # draw a line from DOF origin along the DOF axis
                p1 = frame.origin
                p2 = frame.origin + frame.axis * 0.7
                color = gymapi.Vec3(1.0, 0.0, 0.0)
                gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])


            rigid_contacts = gym.get_env_rigid_contacts(envs[i])
            # 'env0', 'env1', 'body0', 'body1', 'localPos0', 'localPos1', 'minDist', 'initialOverlap', 'normal', 'offset0', 'offset1', 'lambda', 'lambdaFriction', 'friction', 'torsionFriction', 'rollingFriction'
            # if len(rigid_contacts)>0:
            #     print(rigid_contacts[['body0','body1','initialOverlap']])

            gym.draw_env_rigid_contacts(viewer,envs[i],gymapi.Vec3(0.0, 1.0, 1.0),1,True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        # gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

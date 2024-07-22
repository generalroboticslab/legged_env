
import os
# os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib"
# print(os.environ['LD_LIBRARY_PATH'])

import numpy as np
from isaacgym import gymapi, gymutil
import time
import sys
import torch
from isaacgym import gymtorch
from isaacgym import gymapi

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from envs.common.publisher import DataReceiver

# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

asset_descriptors = [   
    # AssetDesc("urdf/v6_2/v6.urdf"),
    # AssetDesc("urdf/biped_visual/biped_v6.urdf", False),
    AssetDesc("urdf/biped/biped_v6.urdf", False),
    AssetDesc("urdf/RobotDog/RobotDog3kg.urdf", False),
    AssetDesc("urdf/a1/a1_minimum.urdf", False),
    AssetDesc("urdf/a1/a1_minimum_anymal_like.urdf", False),
    AssetDesc("urdf/anymal_c/urdf/anymal.urdf", True),
    AssetDesc("urdf/anymal_c/urdf/anymal_minimal.urdf", True),
    AssetDesc("urdf/anymal_c/urdf/anymal_minimal_a1_like.urdf", True),
]


class Visualizer:

    def __init__(self, asset_id):
        self.asset_id = asset_id


        self.receiver = DataReceiver(decoding="msgpack")
        self.receiver.receive_continuously()


        compute_device_id = 0
        graphics_device_id = 0
        physics_engine = gymapi.SIM_PHYSX

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.005
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0,0,-10)

        sim_params.use_gpu_pipeline = False  #  Forcing CPU pipeline
        sim_params.substeps = 1
        sim_params.physx.solver_type = 0
        sim_params.physx.num_position_iterations = 1
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.use_gpu = False
        sim_params.physx.contact_offset=0.02
        sim_params.physx.bounce_threshold_velocity = 0.2
        # sim_params.physx.rest_offset=0.0 # this seems to not work
        sim_params.physx.num_threads=1


        sim_params.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP
        #CC_NEVER : Don’t collect any contacts (value = 0).
        # CC_LAST_SUBSTEP : Collect contacts for last substep only (value = 1).
        # CC_ALL_SUBSTEPS : Collect contacts for all substeps (value = 2) (default).


        # set up the env grid
        num_per_row = 1
        spacing = 1.
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # load asset
        asset_root = os.path.abspath(os.path.dirname(__file__))
        asset_file = asset_descriptors[0].file_name

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = asset_descriptors[0].flip_visual_attachments
        asset_options.use_mesh_materials = True
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = False
        asset_options.override_inertia = False
        asset_options.vhacd_enabled = False
        asset_options.disable_gravity = False
        asset_options.enable_gyroscopic_forces=False

        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0 # DOF armature, a value added to the diagonal of the joint-space inertia matrix. Physically, it corresponds to the rotating part of a motor - which increases the inertia of the joint, even when the rigid bodies connected by the joint can have very little inertia.
        asset_options.thickness = 0.01


        # HACK: only set this for visualization and collision checking
        sim_params.dt = 1e-3
        sim_params.physx.contact_offset=0.0
        sim_params.physx.rest_offset=0.0
        # HACK only set this to prevent robot from moving
        asset_options.fix_base_link = True
        # asset_options.angular_damping = 100.0
        # asset_options.linear_damping = 100.0
        asset_options.disable_gravity = True # False


        # initialize gym
        self.gym = gymapi.acquire_gym()


        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # create viewer
        self.camera_properties = gymapi.CameraProperties()
        # camera_properties.supersampling_horizontal = 2
        # camera_properties.supersampling_vertical = 2

        self.viewer = self.gym.create_viewer(self.sim, self.camera_properties)
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()


        # position the camera
        cam_pos = gymapi.Vec3(0, -0.5, 1.3)
        cam_target = gymapi.Vec3(0, 0, 1.15)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


        #------------------------------------------------------------
        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # get array of DOF names
        self.dof_names = self.gym.get_asset_dof_names(self.asset)

        # get array of DOF properties
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)

        # Maps rigid body names to asset-relative indices
        rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset)

        # Maps joint names to asset-relative indices
        joint_dict = self.gym.get_asset_joint_dict(self.asset)

        # create an array of DOF states that will be used to update the actors
        self.num_dofs = self.gym.get_asset_dof_count(self.asset)
        self.dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
        

        # # self.dof_props['stiffness'][:]=100 # HACK
        self.dof_props['damping'][:] = 100
        self.dof_props['friction'][:] = 100

        # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        # create env
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.32)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1) #xyzw

        self.actor_handle = self.gym.create_actor(self.env, self.asset, pose, "actor", 0, 0)

        self.gym.set_actor_dof_properties(self.env,self.actor_handle,self.dof_props)

        # set default DOF positions
        self.gym.set_actor_dof_states(self.env, self.actor_handle, self.dof_states, gymapi.STATE_ALL)

        #-------------------
        self.rigid_body_names =  self.gym.get_actor_rigid_body_names(self.env,self.actor_handle)
        self.rigid_body_properties = self.gym.get_actor_rigid_body_properties(self.env,self.actor_handle)

        


        # -----------------------------------------------------------------------------
        # subscribe to keyboard events
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

        self.should_close = False

        self.print_infos()

        self.last_frame_time = 0


        pos = [0,0,1]
        rot = [0,0,0,1]
        v_lin = [0,0,0]
        v_ang = [0,0,0]
        np.testing.assert_almost_equal(np.square(rot).sum(), 1, decimal=6, err_msg="env.baseInitState.rot should be normalized to 1")
        self.base_init_state = pos + rot + v_lin + v_ang

        env_ids = torch.tensor([0], dtype=torch.int32)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env_ids_raw = gymtorch.unwrap_tensor(env_ids_int32)

        self.root_state_raw = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(self.root_state_raw)
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float32)

        self.root_state[env_ids] = self.base_init_state


        while not self.should_close:
            self.should_close = self.gym.query_viewer_has_closed(self.viewer)
            # Get input actions from the viewer and handle them appropriately
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "reset" and evt.value > 0:
                    print("reset pressed")

            if self.receiver.data is not None:
                if "real" in self.receiver.data:
                    # print(self.receiver.data["dof_pos"][0])
                    self.dof_states['pos'][:] = np.array(self.receiver.data["real"]["dof_pos"])
                    # print(self.receiver.data["dof_vel"][0])
                    self.dof_states['vel'][:] = np.array(self.receiver.data["real"]["dof_vel"])
                    # print(self.dof_states['pos'])
                    self.gym.set_actor_dof_states(self.env, self.actor_handle, self.dof_states, gymapi.STATE_ALL)
                    if "base_quat" in self.receiver.data["real"]:
                        self.root_state[0,3:7] = torch.tensor(self.receiver.data["real"]["base_quat"], dtype=torch.float32)
                        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_state_raw, env_ids_raw, len(env_ids))

            time.sleep(0.002) # HACK SOMEHOW NEED THIS FOR SIMULATION TO TRACK CORRECTLY

            # dof_states['pos'][:] = np.array([0.000,0.175,0.000,0.387,-0.213,0.000,-0.175,0.000,-0.387,0.213])
            # self.gym.set_actor_dof_states(self.env, self.actor_handle, self.dof_states, gymapi.STATE_ALL)

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.gym.clear_lines(self.viewer)

            # clone actor state in all of the environments
            # gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_POS)
            #show all axis:
            for k in range(self.num_dofs):
                # get the DOF frame (origin and axis)
                dof_handle = self.gym.get_actor_dof_handle(self.env, self.actor_handle, k)
                frame = self.gym.get_dof_frame(self.env, dof_handle)

                # draw a line from DOF origin along the DOF axis
                p1 = frame.origin
                p2 = frame.origin + frame.axis * 0.5
                color = gymapi.Vec3(1.0, 0.0, 0.0)
                gymutil.draw_line(p1, p2, color, self.gym, self.viewer, self.env)


                rigid_contacts = self.gym.get_env_rigid_contacts(self.env)
                # print(rigid_contacts)
                # 'env0', 'env1', 'body0', 'body1', 'localPos0', 'localPos1', 'minDist', 'initialOverlap', 'normal', 'offset0', 'offset1', 'lambda', 'lambdaFriction', 'friction', 'torsionFriction', 'rollingFriction'
                # if len(rigid_contacts)>0:
                #     print(rigid_contacts[['body0','body1','initialOverlap']])

                self.gym.draw_env_rigid_contacts(self.viewer,self.env,gymapi.Vec3(0.0, 1.0, 1.0),1,True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

            delta = time.time() - self.last_frame_time
            if delta < 0.01:
                time.sleep(0.01 - delta)
            self.last_frame_time = time.time()

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


    def print_infos(self):
        # Print DOF properties

        # get list of DOF types
        dof_types = [self.gym.get_asset_dof_type(self.asset, i) for i in range(self.num_dofs)]

        # get the position slice of the DOF state array
        self.dof_positions = self.dof_states['pos']

        # get the limit-related slices of the DOF properties array
        has_limits = self.dof_props['hasLimits']

        for i in range(self.num_dofs):
            print("DOF %d" % i)
            print("  Name:     '%s'" % self.dof_names[i])
            print("  Type:     %s" % self.gym.get_dof_type_string(dof_types[i]))
            print("  Stiffness:  %r" % self.dof_props['stiffness'][i])
            print("  Damping:  %r" % self.dof_props['damping'][i])
            print("  Armature:  %r" % self.dof_props['armature'][i])
            print("  Limited?  %r" % has_limits[i])
            if has_limits[i]:
                print("    Lower   %f" % self.dof_props['lower'][i])
                print("    Upper   %f" % self.dof_props['upper'][i])

        rigid_body_masses = [p.mass for p in self.rigid_body_properties]
        rigid_body_inertias = [p.inertia for p in self.rigid_body_properties]

        total_mass = np.sum(rigid_body_masses)
        for name,mass in zip(self.rigid_body_names,rigid_body_masses):
            print(f"{name:20s},{mass:.5f}")
        print(f"total mass: {total_mass} kg")

        print(f"{'name':20s}:{'i.x.x':<11s},{'i.x.y':<11s},{'i.x.z':<11s},{'i.y.y':<11s},{'i.y.z':<11s},{'i.z.z':<11s}")
        for name,i in zip(self.rigid_body_names,rigid_body_inertias):
            print(f"{name:20s}:{i.x.x:<+8.4e},{i.x.y:<+8.4e},{i.x.z:<+8.4e},{i.y.y:<+8.4e},{i.y.z:<+8.4e},{i.z.z:<+8.4e}")

visualizer = Visualizer(asset_id=0)
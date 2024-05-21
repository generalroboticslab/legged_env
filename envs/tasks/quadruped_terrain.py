# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import copy
import functools
from typing import Dict, Any, Tuple
from operator import inv, itemgetter
from gym import spaces

from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
from typing import Tuple, Dict
from envs.common.utils import bcolors as bc
from envs.common.publisher import DataPublisher
from envs.common.terrrain import Terrain
from isaacgym import gymutil


class QuadrupedTerrain(VecTask):
    """
    issaac gym envs for task "A1Terrain" and "AnymalTerrain"
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.viewer_follow = self.cfg["env"]["viewer"]["follow"]
        self.viewer_follow_offset=torch.tensor(self.cfg["env"]["viewer"].get("follower_offset",[0.5,0.5,0.5]))
        
        self.init_done = False
        infer_observation = self.cfg["env"]["numObservations"] == "infer"
        if infer_observation:
            self.cfg["env"]["numObservations"] = 0
            
        infer_action = self.cfg["env"]["numActions"] == "infer"
        if infer_action:
            self.cfg["env"]["numActions"] = len(self.cfg["env"]["defaultJointAngles"])

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.heightmap_scale = self.cfg["env"]["learn"]["heightMapScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        
        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        
        np.testing.assert_almost_equal(np.square(rot).sum(),1,decimal=6,err_msg="env.baseInitState.rot square sum should be close to 1")
        
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang
        
        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.dt_inv = 1.0 / self.dt
        self.rl_dt = self.dt*self.decimation
        self.rl_dt_inv = 1.0 / self.rl_dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.kp = self.cfg["env"]["control"]["stiffness"]
        self.kd = self.cfg["env"]["control"]["damping"]
        self.torque_limit = self.cfg["env"]["control"]["limit"]  # Torque limit [N.m]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.enable_udp = self.cfg["env"]["enableUDP"]

        # reward scales
        cfg_reward = self.cfg["env"]["learn"]["reward"]
        
        self.rew_scales = {
            "termination": self.cfg["env"]["learn"]["terminalReward"], # TODO. CHANGE THIS
            "lin_vel_xy": cfg_reward["linearVelocityXY"]["scale"],
            "lin_vel_z": cfg_reward["linearVelocityZ"]["scale"],
            "ang_vel_z": cfg_reward["angularVelocityZ"]["scale"],
            "ang_vel_xy": cfg_reward["angularVelocityXY"]["scale"],
            "orient": cfg_reward["orientation"]["scale"],
            "torque": cfg_reward["torque"]["scale"],
            "joint_acc": cfg_reward["jointAcc"]["scale"],
            "joint_vel": cfg_reward["jointVel"]["scale"],
            "joint_pos": cfg_reward["jointPos"]["scale"],
            "joint_pow": cfg_reward["jointPow"]["scale"],
            "base_height": cfg_reward["baseHeight"]["scale"],
            "air_time": cfg_reward["feetAirTime"]["scale"],
            "stance_time": cfg_reward["feetStanceTime"]["scale"],
            # "contact": cfg_reward["feetContact"]["scale"],
            "collision": cfg_reward["kneeCollision"]["scale"],
            "impact": cfg_reward["feetImpact"]["scale"],
            "stumble": cfg_reward["feetStumble"]["scale"],
            "slip": cfg_reward["feetSlip"]["scale"],
            "action": cfg_reward["action"]["scale"],
            "action_rate": cfg_reward["actionRate"]["scale"],
            "hip": cfg_reward["hip"]["scale"],
            "dof_limit": cfg_reward["dofLimit"]["scale"], # TODO CHANGE THIS
            "contact_force": cfg_reward["feetContactForce"]["scale"],
        }
        
        
        
        for key in self.rew_scales.keys():
            self.rew_scales[key] = float(self.rew_scales[key]) * self.rl_dt
        self.torque_penalty_bound = self.cfg["env"]["learn"].get("torquePenaltyBound", 0.0)
        print(f"torque penalty bound = {self.torque_penalty_bound}")

        self.max_feet_contact_force = 100  # [N] # todo refactor

        # base height reward: reverse bell shaped curve
        # https://researchhubs.com/post/maths/fundamentals/bell-shaped-function.html
        a, b = self.cfg["env"]["learn"].get("baseHeightRewardParams", [0.04, 3])
        self.base_height_rew_a, self.base_height_rew_b = float(a), float(b)

        # min air time and stance time in seconds
        self.air_time_offset = float(cfg_reward["feetAirTime"]["offset"])
        self.stance_time_offset = float(cfg_reward["feetStanceTime"]["offset"])

        # treat commends below this threshold as zero [m/s]
        self.command_zero_threshold = self.cfg["env"]["commandZeroThreshold"]

        # target base height [m]
        self.target_base_height = self.cfg["env"]["baseHeightTarget"]

        # default joint positions [rad]
        self.named_default_joint_pos = self.cfg["env"]["defaultJointAngles"]

        # desired joint positions [rad]
        self.named_desired_joint_pos = self.cfg["env"]["desiredJointAngles"]

        self.num_environments = self.cfg["env"]["numEnvs"]

        # heightmap
        self.init_height_points()  # height_points in cpu

        self.obs_names = tuple(self.cfg["env"]["observationNames"])

        self.num_dof = len(cfg["env"]["defaultJointAngles"])  # 12, todos, remove duplicates ,search self.num_dof
        # observation dimensions of specific items
        self.obs_dim_dict = {
            "linearVelocity": 3,
            "angularVelocity": 3,
            "projectedGravity": 3,
            "commands": 3,  # vel_x,vel_y, vel_yaw, (excluding heading)
            "dofPosition": self.num_dof,
            "dofVelocity": self.num_dof,
            "heightMap": self.num_height_points - 1,  # excluding the base origin measuring point
            "actions": self.num_dof,
            "contact": 4,  # feet contact indicator
        }

        num_obs = np.sum(itemgetter(*self.obs_names)(self.obs_dim_dict))
        if infer_observation:
            # TODO refactor this number to be automatic
            self.cfg["env"]["numObservations"] = self.num_observations = num_obs
            # self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
            # self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
            print(f"inferring, numObservations={num_obs}")
        assert self.cfg["env"]["numObservations"] == num_obs
        
        

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # height points to device
        self.height_points = self.height_points.to(self.device)

        # push robot
        self.should_push_robots = self.cfg["env"]["learn"]["pushRobots"]
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.push_vel_min = torch.tensor(self.cfg["env"]["learn"]["pushVelMin"], dtype=torch.float, device=self.device)
        self.push_vel_max = torch.tensor(self.cfg["env"]["learn"]["pushVelMax"], dtype=torch.float, device=self.device)

        # get gym GPU state tensors
        self.root_state_raw = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_raw = self.gym.acquire_dof_state_tensor(self.sim)
        self.rb_state_raw = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_state = gymtorch.wrap_tensor(self.root_state_raw)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_raw)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        # rb_state: (num_envs,num_rigid_bodies,13)
        self.rb_state = gymtorch.wrap_tensor(self.rb_state_raw).view(self.num_envs, -1, 13)
        # contact_forces: (num_envs, num_bodies, xyz axis)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.noise_vec = torch.zeros_like(self.obs_buf, dtype=torch.float, device=self.device)
        # commands: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], dtype=torch.float, device=self.device
        )
        # gravity_vec=[0,0,-1]
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.torque = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        # self.actions_filt = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        # self.dof_pos_filt = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)  # feet air time
        self.stance_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)  # feet stance time
        self.last_feet_contact_force = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        # joint positions offsets
        # self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device)
        # for i in range(self.num_actions):
        #     name = self.dof_names[i]
        #     angle = self.named_default_joint_angles[name]
        #     self.default_dof_pos[:, i] = angle
        self.default_dof_pos = torch.tensor(
            itemgetter(*self.dof_names)(self.named_default_joint_pos), dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        self.desired_dof_pos = torch.tensor(
            itemgetter(*self.dof_names)(self.named_desired_joint_pos), dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        # reward episode sums (unscaled)
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_sums = {key: torch_zeros() for key in self.rew_scales.keys()}

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True
        
        # self.enable_viewer_sync = False  # by default freeze the viewer until "V" is pressed
        
        if self.graphics_device_id != -1 and self.viewer:
            
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_F, "toggle_viewer_follow")
                        
            self.cam_pos = self.cfg["env"]["viewer"]["pos"]
            self.cam_target_pos = self.cfg["env"]["viewer"]["lookat"]
            if self.viewer_follow:
                self.cam_target_pos= self.root_state[0, :3].clone().cpu() 
                self.cam_pos= self.viewer_follow_offset.cpu() +self.cam_target_pos
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(*self.cam_pos), gymapi.Vec3(*self.cam_target_pos))
            
        if self.enable_udp:  # plotJuggler related
            self.t = 0
            self.data_publisher = DataPublisher(is_enabled=True)

    def process_keyboard_events(self,event):
        if event.action == "toggle_viewer_follow" and event.value > 0:
            self.viewer_follow = not self.viewer_follow
    
    
    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.terrain_type = self.cfg["env"]["terrain"]["terrainType"]

        # create plane/triangle mesh/heigh field
        self.terrain = Terrain(
            self.cfg["env"]["terrain"], num_robots=self.num_envs, device=self.device, gym=self.gym, sim=self.sim
        )
        if self.terrain_type in {'trimesh', 'heightfiled'}:
            self.custom_origins = True

        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _get_noise_scale_vec(self):

        cfg_learn = self.cfg["env"]["learn"]
        self.add_noise = cfg_learn["addNoise"]
        noise_level = cfg_learn["noiseLevel"]
        noise_dict = {
            "linearVelocity": cfg_learn["linearVelocityNoise"] * noise_level * self.lin_vel_scale,
            "angularVelocity": cfg_learn["angularVelocityNoise"] * noise_level * self.ang_vel_scale,
            "projectedGravity": cfg_learn["gravityNoise"] * noise_level,
            "commands": 0,
            "dofPosition": cfg_learn["dofPositionNoise"] * noise_level * self.dof_pos_scale,
            "dofVelocity": cfg_learn["dofVelocityNoise"] * noise_level * self.dof_vel_scale,
            "heightMap": cfg_learn["heightMapNoise"] * noise_level * self.heightmap_scale,
            "actions": 0,  # previous actions
            "contact": 0,  # feet contact
        }
        noise_vec_lists = [torch.ones(self.obs_dim_dict[name]) * noise_dict[name] for name in self.obs_names]
        noise_vec = torch.cat(noise_vec_lists, dim=-1).to(self.device)
        return noise_vec

    # def load_asset(self):
        
        
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        cfg_asset = self.cfg["env"]["urdfAsset"]
        asset_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        if "root" in cfg_asset:
            asset_root = cfg_asset["root"]
        asset_file = cfg_asset["file"]
        # body names
        base_name = cfg_asset["baseName"]
        hip_name = cfg_asset["hipName"]
        foot_name = cfg_asset["footName"]
        knee_name = cfg_asset["kneeName"]
        # joint names
        hip_joint_name = cfg_asset["hipJointName"]

        # bitwise filter for elements in the same collisionGroup to mask off collision
        self.collision_filter = cfg_asset["collision_filter"]

        asset_options = gymapi.AssetOptions()
        # defaults
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        for attribute in cfg_asset["AssetOptions"]:
            if hasattr(asset_options, attribute):
                setattr(asset_options, attribute, cfg_asset["AssetOptions"][attribute])
            else:
                print(f"{bc.WARNING}{attribute} not in AssetOptions!{bc.ENDC}")

        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)


        body_names = self.gym.get_asset_rigid_body_names(self.asset)
        self.dof_names = self.gym.get_asset_dof_names(self.asset)


        def get_matching_str(source, destination,comment=""):
            """Finds case-insensitive partial matches between source and destination lists."""
            if type(source) is str: # one to many
                matches = [item for item in destination if source.lower() in item.lower()]  
                if not matches:
                    raise KeyError(f"cannot locate {source} [{comment}\navailables are {destination}")
                return matches
            # one to one     
            def find_matches(src_item):
                matches = [item for item in destination if src_item.lower() in item.lower()]
                if not matches:
                    raise KeyError(f"cannot locate {src_item}. [{comment}]")
                elif len(matches) > 1:
                    raise KeyError(f"find multiple instances for {src_item}. [{comment}]")
                return matches[0]  # Return just the first match
            return [find_matches(item) for item in source]

        
        # body
        base_name = get_matching_str(source=base_name, destination=body_names, comment="base_name")[0]
        hip_names = get_matching_str(source=hip_name, destination=body_names, comment="hip_name")
        feet_names = get_matching_str(source=foot_name, destination=body_names, comment="foot_name")
        knee_names = get_matching_str(source=knee_name, destination=body_names, comment="knee_name")


        ######################
        # base_name = [s for s in body_names if base_name.lower() in s.lower()][0]
        # hip_names = [s for s in body_names if hip_name.lower() in s.lower()]
        # feet_names = [s for s in body_names if foot_name.lower() in s.lower()]
        # knee_names = [s for s in body_names if knee_name.lower() in s.lower()]
        # find_rb_handle = functools.partial(self.gym.find_actor_rigid_body_handle, self.envs[0], self.actor_handles[0])
        # self.base_id = find_rb_handle(base_name)
        # self.hip_ids = torch.tensor([find_rb_handle(n) for n in hip_names], dtype=torch.long, device=self.device)
        # self.knee_ids = torch.tensor([find_rb_handle(n) for n in knee_names], dtype=torch.long, device=self.device)
        # self.feet_ids = torch.tensor([find_rb_handle(n) for n in feet_names], dtype=torch.long, device=self.device)
        
        # # joints 
        # dof_hip_names = get_matching_str(source=hip_joint_name,destination=self.dof_names, comment="hip_joint_name")
        # find_dof_handle = functools.partial(self.gym.find_actor_dof_handle, self.envs[0], self.actor_handles[0])
        # self.dof_hip_ids = torch.tensor(
        #     [find_dof_handle(n) for n in dof_hip_names], dtype=torch.long, device=self.device
        # )
        
        asset_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset)
        self.base_id = asset_rigid_body_dict[base_name]
        self.hip_ids = torch.tensor([asset_rigid_body_dict[n] for n in hip_names], dtype=torch.long, device=self.device)
        self.knee_ids = torch.tensor([asset_rigid_body_dict[n] for n in knee_names], dtype=torch.long, device=self.device)
        self.feet_ids = torch.tensor([asset_rigid_body_dict[n] for n in feet_names], dtype=torch.long, device=self.device)

        # joints
        dof_hip_names = get_matching_str(source=hip_joint_name,destination=self.dof_names, comment="hip_joint_name")
        asset_dof_dict = self.gym.get_asset_dof_dict(self.asset)
        self.dof_hip_ids = torch.tensor([asset_dof_dict[n] for n in dof_hip_names], dtype=torch.long, device=self.device)
        

        print(f"base = {base_name}: {self.base_id}")
        print(f"hip = {dict(zip(hip_names,self.hip_ids.tolist()))}")
        print(f"knee = {dict(zip(knee_names,self.knee_ids.tolist()))}")
        print(f"feet = {dict(zip(feet_names,self.feet_ids.tolist()))}")
        print(f"dof_hip = {dict(zip(dof_hip_names,self.dof_hip_ids.tolist()))}")
        assert self.base_id!=-1
        assert len(self.dof_hip_ids)==4
        # assert self.dof_hip_ids.tolist() == [0, 3, 6, 9]
        ####################

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(self.asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        # num_buckets = 100 # TODO do regression test
        num_buckets = self.num_envs
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), self.device)

        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        # asset dof properties override
        asset_dof_properties = self.cfg["env"].get("assetDofProperties", {})
        if asset_dof_properties is not None:
            for key, value in asset_dof_properties.items():
                self.dof_props[key][:] = value  # used in set_actor_dof_properties
                print(f"overwrite asset dof [{key}]: {value}")

        # dof limit
        dof_margin = self.cfg["env"]["learn"].get("dofLimitMargins", 0)
        dof_margin = torch.tensor(dof_margin, dtype=torch.float, device=self.device)
        self.dof_lower = torch.tensor(self.dof_props['lower'], dtype=torch.float, device=self.device) + dof_margin
        self.dof_upper = torch.tensor(self.dof_props['upper'], dtype=torch.float, device=self.device) - dof_margin

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        cfg_terrain = self.cfg["env"]["terrain"]
        if not self.curriculum:
            cfg_terrain["maxInitMapLevel"] = cfg_terrain["numLevels"] - 1
        self.terrain_levels = torch.randint(0, cfg_terrain["maxInitMapLevel"] + 1, (self.num_envs,), device=self.device)
        self.terrain_level_mean = self.terrain_levels.float().mean()
        self.heights_curriculum_started = False
        self.heights_curriculum_ratio = 0.001
        self.terrain_types = torch.randint(0, cfg_terrain["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            spacing = 0.0
            self.base_pos_xy_range = (-1.0, 1.0)  # TODO refactor into config

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain.env_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(*self.base_pos_xy_range, (2, 1), self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            self.gym.set_asset_rigid_shape_properties(self.asset, rigid_shape_prop)
            actor_handle = self.gym.create_actor(
                env_handle, self.asset, start_pose, "actor", i, self.collision_filter, 0
            )
            self.gym.set_actor_dof_properties(env_handle, actor_handle, self.dof_props)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        if self.cfg["env"]["learn"]["randomizeBaseMass"]:
            added_masses = np.random.uniform(*self.cfg["env"]["learn"]["addedMassRange"], self.num_envs)
            for i in range(self.num_envs):
                env_handle = self.envs[i]
                actor_handle = self.actor_handles[i]
                body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
                body_props[self.base_id].mass += added_masses[i]
                self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

    def check_termination(self):
        self.reset_buf = torch.norm(self.contact_forces[:, self.base_id, :], dim=1) > 1.0
        if not self.allow_knee_contacts:
            knee_contact = torch.norm(self.contact_forces[:, self.knee_ids, :], dim=2) > 1.0
            self.reset_buf |= torch.any(knee_contact, dim=1)
        self.reset_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def compute_observations(self):

        self.get_heights()
        heights = torch.clip(self.heights_relative - self.target_base_height, -1.0, 1.0) * self.heightmap_scale

        obs_dict = {
            "linearVelocity": self.base_lin_vel * self.lin_vel_scale,
            "angularVelocity": self.base_ang_vel * self.ang_vel_scale,
            "projectedGravity": self.projected_gravity,
            "commands": self.commands[:, :3] * self.commands_scale,
            "dofPosition": self.dof_pos * self.dof_pos_scale,
            "dofVelocity": self.dof_vel * self.dof_vel_scale,
            "heightMap": heights[:, :-1],
            "actions": self.actions,
            "contact": self.feet_contact,
        }
        self.obs_buf = torch.cat(itemgetter(*self.obs_names)(obs_dict), dim=-1)

        if self.add_noise:
            self.noise_vec.uniform_(-1.0, 1.0).mul_(self.noise_scale_vec)  # scaled noise vector
            self.obs_buf += self.noise_vec

    def compute_reward(self):
        rew = {}
        # velocity tracking reward
        lin_vel_error = square_sum(self.commands[:, :2] - self.base_lin_vel[:, :2])
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew["lin_vel_xy"] = torch.exp(-lin_vel_error / 0.25)
        rew["ang_vel_z"] = torch.exp(-ang_vel_error / 0.25)

        # other base velocity penalties
        rew["lin_vel_z"] = torch.square(self.base_lin_vel[:, 2])
        rew["ang_vel_xy"] = square_sum(self.base_ang_vel[:, :2])

        # orientation penalty
        rew["orient"] = square_sum(self.projected_gravity[:, :2])

        # base height penalty
        # rew["base_height"] = torch.square(self.heights_relative[:, -1] - self.target_base_height)
        # exponential reward
        # base_height_error = torch.square(self.heights_relative[:, -1] - self.target_base_height)
        # rew["base_height"] = 1.0 - torch.exp(-base_height_error / 0.001)
        # bell-shaped reward
        rew["base_height"] = reverse_bell(
            self.heights_relative[:, -1], a=self.base_height_rew_a, b=self.base_height_rew_b, c=self.target_base_height
        )

        # torque penalty
        rew["torque"] = square_sum(self.torque)
        
        # rew["torque"] = out_of_float_bound_squared_sum(
        #     self.torque, -self.torque_penalty_bound, self.torque_penalty_bound
        # )
        # rew["torque"] = out_of_bound_norm(self.torque, -self.torque_penalty_bound, self.torque_penalty_bound)

        # joint acc penalty
        # rew["joint_acc"] = torch.square(self.last_dof_vel - self.dof_vel).sum(dim=1)
        dof_acc = (self.dof_vel - self.last_dof_vel) * self.rl_dt_inv  # TODO check if [:] is needed # TODO
        rew["joint_acc"] = square_sum(dof_acc)  # TODO Check this
        # rew["joint_acc"] = torch.abs(dof_acc).sum(dim=1)  #TODO Check this

        # joint vel penalty
        rew["joint_vel"] = square_sum(self.dof_vel)


        # joint position penalty
        # rew["joint_pos"] = (self.dof_pos - self.default_dof_pos).abs().sum(dim=1)
        rew["joint_pos"] = (self.dof_pos - self.desired_dof_pos).abs().sum(dim=1)
        # rew["joint_pos"] = (self.dof_pos_filt - self.desired_dof_pos).abs().sum(dim=1)
        
        # joint power penalty
        rew["joint_pow"] = (self.dof_vel * self.torque).abs().sum(dim=1)

        # collision penalty
        knee_collision = torch.norm(self.contact_forces[:, self.knee_ids, :], dim=2) > 1.0
        rew["collision"] = torch.sum(knee_collision, dim=1, dtype=torch.float)  # sum vs any ?

        # feet impact penalty (num_envs,4,3)
        feet_contact_diff = self.feet_contact_force[:, :, 2] - self.last_feet_contact_force[:, :, 2]
        rew["impact"] = feet_contact_diff.abs().sum(dim=1)
        # rew["impact"] = feet_contact_diff.view(self.num_envs,-1).abs().sum(dim=1)
        # feet_contact_diff = self.feet_contact_force - self.last_feet_contact_force
        # rew["impact"] = torch.norm(feet_contact_diff,dim=2).sum(dim=1)

        # stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.feet_ids, :2], dim=2) > 5.0) * (
            torch.abs(self.contact_forces[:, self.feet_ids, 2]) < 1.0
        )
        rew["stumble"] = torch.sum(stumble, dim=1, dtype=torch.float)

        # feet slip penalty
        rew["slip"] = (self.feet_lin_vel.square().sum(dim=2) * self.feet_contact).sum(dim=1)

        # action penalty
        # rew["action"] = torch.square(self.actions).sum(dim=1)
        rew["action"] = self.actions.abs().sum(dim=1)
        # rew["action"] = self.actions_filt.abs().sum(dim=1)  # filtered

        # action rate penalty
        # rew["action_rate"] = torch.square(self.last_actions - self.actions).sum(dim=1)
        action_rate = (self.actions - self.last_actions) * self.rl_dt_inv
        rew["action_rate"] = torch.square(action_rate).sum(dim=1)
        # rew["action_rate"] = torch.abs((self.last_actions - self.actions)*self.dt_inv).sum(dim=1)

        nonzero_command = torch.norm(self.commands[:, :2], dim=1) > self.command_zero_threshold
        feet_no_contact = ~self.feet_contact
        # air time reward (reward long swing)
        first_contact = (self.air_time > 0.0) * self.feet_contact
        self.air_time += self.rl_dt
        # reward only on first contact with the ground
        rew["air_time"] = (
            torch.sum((self.air_time + self.air_time_offset) * first_contact, dim=1, dtype=torch.float)
            * nonzero_command  # no reward for zero command
        )
        self.air_time *= feet_no_contact  # reset if contact

        # feet stance time reward (reward long stance)
        first_no_contact = (self.stance_time > 0.0) * feet_no_contact
        self.stance_time += self.rl_dt
        # reward only on first leaving the ground
        rew["stance_time"] = (
            torch.sum((self.stance_time + self.stance_time_offset) * first_no_contact, dim=1, dtype=torch.float)
            * nonzero_command  # no reward for zero command
        )
        self.stance_time *= self.feet_contact  # reset if no contact

        # # reward contact at zero command
        # rew["contact"] = torch.sum(self.feet_contact, dim=1,dtype=torch.float)* (~nonzero_command)

        # penalize high contact forces
        contact_force_norm = torch.norm(self.contact_forces[:, self.feet_ids, :], dim=-1)
        rew["contact_force"] = torch.sum((contact_force_norm - self.max_feet_contact_force).clip(min=0.0), dim=1)

        # cosmetic penalty for hip motion
        # rew["hip"] = torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        # rew["hip"] = (self.dof_pos[:, self.dof_hip_ids] - self.default_dof_pos[:, self.dof_hip_ids]).abs().sum(dim=1)
        # rew["hip"] =(self.dof_pos - self.default_dof_pos).abs().sum(dim=1)
        rew["hip"] = (self.dof_pos[:, self.dof_hip_ids] - self.desired_dof_pos[:, self.dof_hip_ids]).abs().sum(dim=1)

        # penalty for position exceeding dof limit
        # rew["dof_limit"] = out_of_bound_norm(self.dof_pos, self.dof_lower, self.dof_upper)
        rew["dof_limit"] = out_of_bound_abs_sum(self.dof_pos, self.dof_lower, self.dof_upper)

        # log episode reward sums
        for key in rew.keys():
            self.episode_sums[key] += rew[key]  # unscaled
            rew[key] *= self.rew_scales[key]
        # total reward
        self.rew_buf = (
            rew["lin_vel_xy"]
            + rew["ang_vel_z"]
            + rew["lin_vel_z"]
            + rew["ang_vel_xy"]
            + rew["orient"]
            + rew["base_height"]
            + rew["torque"]
            + rew["joint_acc"]
            + rew["joint_vel"]
            + rew["joint_pos"]
            + rew["joint_pow"]
            + rew["collision"]
            + rew["impact"]
            + rew["stumble"]
            + rew["slip"]
            + rew["action"]
            + rew["action_rate"]
            + rew["air_time"]
            + rew["stance_time"]
            # + rew["contact"]
            + rew["contact_force"]
            + rew["hip"]
            + rew["dof_limit"]
        )
        self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        if self.enable_udp:  # send UDP info to plotjuggler
            self.t += self.dt

            foot_pos_rel = self.rb_state[:, self.feet_ids, 0:3] - self.root_state[:, :3].view(self.num_envs, 1, 3)
            foot_pos = quat_rotate_inverse(self.base_quat.repeat_interleave(4, dim=0), foot_pos_rel.view(-1, 3)).view(
                self.num_envs, 4, 3
            )
            data = {
                "t": self.t,
                "time_air":self.air_time,
                "time_stance":self.stance_time,
                "foot_pos": foot_pos,
                "base_height": self.heights_relative[:, -1],
                "dof_vel": self.dof_vel,
                "dof_pos": self.dof_pos,
                "dof_pos_target": (self.action_scale * self.actions + self.default_dof_pos),
                "dof_acc": dof_acc,
                "dof_effort": self.torque,
                "base_lin_vel": self.base_lin_vel,
                "base_ang_vel": self.base_ang_vel,
                "action": self.actions,
                "action_rate": action_rate,
                "contact": self.feet_contact,
                "rew_buf":self.rew_buf * self.rl_dt_inv
            }
            for key in rew.keys():
                data[f"rew_{key}"] = rew[key] * self.rl_dt_inv
            self.data_publisher.publish(data)

    def reset_idx(self, env_ids):
        len_ids = len(env_ids)
        if len_ids == 0:
            return
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env_ids_raw = gymtorch.unwrap_tensor(env_ids_int32)
        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_state[env_ids] = self.base_init_state
            self.root_state[env_ids, :3] += self.env_origins[env_ids]
            self.root_state[env_ids, :2] += torch_rand_float(*self.base_pos_xy_range, (len_ids, 2), self.device)
        else:
            self.root_state[env_ids] = self.base_init_state
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_state_raw, env_ids_raw, len_ids)

        # positions_offset = torch_rand_float(0.5, 1.5, (len_ids, self.num_dof), self.device)
        dof_pos_offset = torch_rand_float(-0.1, 0.1, (len_ids, self.num_dof), self.device)
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] + dof_pos_offset
        self.dof_vel[env_ids] = torch_rand_float(-0.1, 0.1, (len_ids, self.num_dof), self.device)
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_raw, env_ids_raw, len_ids)

        self.commands[env_ids, 0] = torch_rand_float(*self.command_x_range, (len_ids, 1), self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(*self.command_y_range, (len_ids, 1), self.device).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(*self.command_yaw_range, (len_ids, 1), self.device).squeeze()

        # # set small commands to zero # TODO CHANGE BACK
        # self.commands[env_ids] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > self.command_zero_threshold
        # ).unsqueeze(1)

        self.last_actions[env_ids] = 0.0
        # self.actions_filt[env_ids] = 0.0
        # self.dof_pos_filt[env_ids] = self.dof_pos[env_ids]
        self.last_feet_contact_force[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.stance_time[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            raw_sum = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s  # rewards per second
            self.extras["episode"][f'rew_{key}_raw'] = raw_sum * self.dt  # scaled by policy dt
            self.extras["episode"][f'rew_{key}'] = raw_sum * self.rew_scales[key]
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = self.terrain_level_mean
        self.extras["episode"]["heights_curriculum_ratio"] = self.heights_curriculum_ratio

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_state[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        command_distance = torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s
        self.terrain_levels[env_ids] -= 1 * (distance < command_distance * 0.25)
        # TODO check level up/down condition
        # self.terrain_levels[env_ids] += 1 * (distance > command_distance*0.5)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain.env_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.terrain_level_mean = self.terrain_levels.float().mean()

    def push_robots(self):
        self.root_state[:, 7:13] = torch_rand_tensor(
            self.push_vel_min, self.push_vel_max, (self.num_envs, 6), device=self.device
        )  # lin vel x/y/z
        self.gym.set_actor_root_state_tensor(self.sim, self.root_state_raw)


    def render(self):
        if self.viewer and self.viewer_follow:
            # do modify camera position
            self.cam_target_pos = self.root_state[0, :3].clone().cpu() 
            self.cam_pos = self.viewer_follow_offset.cpu()+self.cam_target_pos
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(*self.cam_pos), gymapi.Vec3(*self.cam_target_pos))     
        super().render()
        return

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        ### TODO currently need to completely bypass this
        # # step physics and render each frame
        # for i in range(self.control_freq_inv):
        #     if self.force_render:
        #         self.render()
        #     self.gym.simulate(self.sim)
        if self.force_render:
            self.render()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        dof_pos_target = self.action_scale * self.actions + self.default_dof_pos
        for i in range(self.decimation):
            torque = torch.clip(
                self.kp * (dof_pos_target - self.dof_pos) - self.kd * self.dof_vel,
                -self.torque_limit,
                self.torque_limit,
            )
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torque))
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.torque = torque.view(self.torque.shape)

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        if self.common_step_counter % self.push_interval == 0 and self.should_push_robots:
            self.push_robots()

        # prepare quantities
        self.base_quat = self.root_state[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
        # feet contact
        # self.contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        self.feet_contact_force = self.contact_forces[:, self.feet_ids, :]
        self.feet_contact = self.feet_contact_force[:, :, 2] > 1.0  # todo check with norm
        self.feet_lin_vel = self.rb_state[:, self.feet_ids, 7:10]

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_observations()
        self.compute_reward()

        # update last_...
        self.last_actions[:] = self.actions[:]
        # self.actions_filt[:] = self.actions_filt * 0.97 + self.actions * 0.03
        # self.dof_pos_filt[:] = self.dof_pos_filt * 0.97 + self.dof_pos * 0.03
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_feet_contact_force[:] = self.feet_contact_force[:]

        # resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            # self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_state[i, :3]).cpu().numpy()
                heights = self.heights_absolute[i].cpu().numpy()
                height_points = (
                    quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                )
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def init_height_points(self):
        """
        initialize height points in cpu, save self.num_height_points and self.height_points
        self.num_height_points[:,:-1,0] is grid_x
        self.num_height_points[:,:-1,1] is grid_y
        self.num_height_points[:,-1,:] is base (0,0,0)
        """
        cfg_heightmap = self.cfg["env"]["heightmap"]
        x = torch.tensor(cfg_heightmap["x"], dtype=torch.float)
        y = torch.tensor(cfg_heightmap["y"], dtype=torch.float)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.num_height_points = grid_x.numel() + 1  # (num_hightmap_points+1)
        points = torch.zeros(self.num_envs, self.num_height_points, 3)
        points[:, :-1, 0] = grid_x.flatten()
        points[:, :-1, 1] = grid_y.flatten()
        # points[:,-1,:] = (0,0,0) # base origin
        self.height_points = points

    def get_heights(self):
        if self.terrain_type == 'plane':
            self.heights_absolute = torch.zeros(self.num_envs, self.num_height_points, device=self.device)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_state[:, :3]
            ).unsqueeze(1)
            # self.root_states: (num_env,13)
            # points: (num_env,num_points_per_env+1 (root_pos),3 (xyz))
            ### points = torch.cat((points, self.root_states[:, :3].unsqueeze(1)), dim=1)
            heights = self.terrain.get_heights(points)
            # heights_absolute: (num_env,num_points_per_env+1 (body_com))
            self.heights_absolute = heights.view(self.num_envs, -1)
        self.heights_relative = self.root_state[:, 2].unsqueeze(1) - self.heights_absolute


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def torch_rand_tensor(lower: torch.Tensor, upper: torch.Tensor, shape: Tuple[int, int], device: str) -> torch.Tensor:
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def square_sum(input: torch.Tensor) -> torch.Tensor:
    return torch.square(input).sum(dim=-1)


@torch.jit.script
def out_of_bound_norm(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.norm(input - torch.clamp(input, lower, upper), dim=-1)


@torch.jit.script
def out_of_bound_abs_sum(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return (input - torch.clamp(input, lower, upper)).abs().sum(dim=-1)

@torch.jit.script
def out_of_float_bound_squared_sum(input: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.square(input - torch.clamp(input, lower, upper)).sum(dim=-1)

# jit is slower here so do not use jit
def abs_sum(input: torch.Tensor) -> torch.Tensor:
    return input.abs().sum(dim=-1)

# https://researchhubs.com/post/maths/fundamentals/bell-shaped-function.html
# https://www.mathworks.com/help/fuzzy/gbellmf.html
# https://www.mathworks.com/help/fuzzy/dsigmf.html
# https://www.mathworks.com/help/fuzzy/foundations-of-fuzzy-logic.html
@torch.jit.script
def bell(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    return 1 / (1 + torch.pow(torch.abs(x / a - c), b))


@torch.jit.script
def reverse_bell(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    return 1 - 1 / (1 + torch.pow(torch.abs(x / a - c), b))

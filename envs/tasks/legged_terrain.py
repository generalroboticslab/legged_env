import numpy as np
import os
import sys
import time
import datetime
from typing import Dict, Any, Tuple, Union
from operator import itemgetter
from gym import spaces
from collections.abc import Iterable

from isaacgym.torch_utils import get_axis_params, torch_rand_float, quat_rotate_inverse, quat_apply, normalize, quat_conjugate,quat_mul
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
from envs.common.utils import bcolors as bc
from envs.common.urdf_utils import get_leaf_nodes, trace_edges, urdf_to_graph
from envs.common.publisher import DataPublisher, DataReceiver
from envs.common.terrain import Terrain
from isaacgym import gymutil

from hydra.utils import to_absolute_path

from envs.common import buffer

class LeggedTerrain(VecTask):
    """
    issaac gym envs any legged robot locomotion task
    """

    def __init__(
            self,
            cfg: Dict[str, Any],
            rl_device: str,
            sim_device: str,
            graphics_device_id: int,
            headless: bool,
            virtual_screen_capture: bool,
            force_render: bool
            ):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array
                                    via `env.render(mode='rgb_array')`.
            force_render: Set to True to always force rendering in the steps
                          (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.init_done = False

        self.cfg = cfg
        self.rl_device = rl_device
        self.headless = headless  # if training in a headless mode
        self.virtual_screen_capture = virtual_screen_capture
        self.force_render = force_render

        # set device and rl_device
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if self.cfg["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() in {"cuda", "gpu"}:
                self.device = f"cuda:{self.device_id}"
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                self.cfg["sim"]["use_gpu_pipeline"] = False

        # Rendering       
        self.graphics_device_id = graphics_device_id 
        enable_camera_sensors = self.cfg["env"].get("enableCameraSensors", False)
        if (not enable_camera_sensors) and self.headless:
            self.graphics_device_id = -1

        self.num_environments = self.cfg["env"]["numEnvs"]  # self.num_envs
        self.num_agents = self.cfg["env"].get("numAgents", 1)  # used for multi-agent environments

        # self.sim_params = self._VecTask__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        self.sim_params = self._parse_sim_params()

        self.gym = gymapi.acquire_gym()

        # Creates the physics simulation and terrain.
        self.up_axis_idx = {"x": 0, "y": 1, "z": 2}[self.cfg["sim"]["up_axis"]]  # index of up axis: x=0, y=1, z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.load_asset()


        # keep still at zero command
        self.keep_still_at_zero_command: bool = self.cfg["env"].get("keep_still_at_zero_command",True)

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        # treat commends below this threshold as zero [m/s]
        # square it to compare the square sum instead of the norm (slightly more efficient)
        self.command_zero_threshold: float = np.square(self.cfg["env"]["commandZeroThreshold"])
        self.command_zero_probability: float = self.cfg["env"]["commandZeroProbability"]
        # TODO group them together in config

        # commands: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], dtype=torch.float, device=self.device
        )
        self.is_zero_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # default joint positions [rad]
        self.named_default_dof_pos = self.cfg["env"].get("defaultJointPositions", {n: 0 for n in self.dof_names})
        # desired joint positions [rad]
        self.named_desired_dof_pos = self.cfg["env"].get("desiredJointPositions", self.named_default_dof_pos)
        # initial joint positions [rad]
        self.named_init_dof_pos = self.cfg["env"].get("initialJointPositions", self.named_desired_dof_pos)

        def get_dof_param(param: Union[float, int, dict, Iterable]) -> torch.Tensor:
            """
            Helper function to get a tensor of parameters for each actuated degree of freedom.
            Args:
                param: parameters, can be a single float/int, a dict, or an iterable
            Returns:
                a tensor of parameters with shape (1, num_actuated_dof)
            """
            if isinstance(param, (float, int)):  # single value
                return param, {n: param for n in self.dof_names}
            elif isinstance(param, dict):  # dict of parameters
                dof_param = np.zeros(self.num_actuated_dof)
                if "default" in param:  # default parameter
                    dof_param[:] = param["default"]
                for dof_name, value in param.items():
                    if dof_name == "default":  # skip the default parameter
                        continue
                    actual_names = get_matching_str(source=dof_name, destination=self.dof_names, case_sensitive=False, comment="")
                    for name in actual_names:  # set the parameter for each matching joint
                        dof_param[self.dof_dict[name]] = value
            elif isinstance(param, Iterable):  # iterable of parameters
                dof_param = np.array(param)
            named_dof_param = {n: p for n,p in zip(self.dof_names,dof_param)}
            dof_param = torch.tensor(dof_param, device=self.device, dtype=torch.float).view(1, -1)  # view as a tensor with shape (1, num_actuated_dof)
            return dof_param,named_dof_param
                
        self.default_dof_pos,self.named_default_dof_pos = get_dof_param(self.named_default_dof_pos)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

        self.desired_dof_pos,self.named_desired_dof_pos = get_dof_param(self.named_desired_dof_pos)
        self.desired_dof_pos = self.desired_dof_pos.repeat(self.num_envs, 1)

        self.init_dof_pos,self.named_init_dof_pos = get_dof_param(self.named_init_dof_pos)
        self.init_dof_pos = self.init_dof_pos.repeat(self.num_envs, 1)

        # control
        self.kp, _ = get_dof_param(self.cfg["env"]["control"]["stiffness"]) # Stiffness [N/m]
        self.kd,_ = get_dof_param(self.cfg["env"]["control"]["damping"]) # Damping [Ns/m]
        self.dof_force_target_limit,_ = get_dof_param(self.cfg["env"]["control"]["limit"]) # Torque limit [N.m]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # TODO: MAYBE SET A BETTER SCALE FOR DOF FORCES
        self.dof_force_scale = self.cfg["env"]["learn"].get("dofForceScale",1/self.dof_force_target_limit)
        self.dof_force_target_scale = self.cfg["env"]["learn"].get("dofForceTargetScale",1/self.dof_force_target_limit)
        self.heightmap_scale = self.cfg["env"]["learn"]["heightMapScale"]
        
        # update pos to default dof pos
        self.asset_urdf.update_cfg(self.named_default_dof_pos)
        lower_bound_z = self.asset_urdf.collision_scene.bounding_box.bounds[0,2]
        
        # target base height [m]
        self.base_height_target = self.cfg["env"].get("baseHeightTarget",None)

        if self.base_height_target is None:
            base_height_offset= self.cfg["env"].get("baseHeightOffset",0.1)
            base_height_tareget_offset = self.cfg["env"].get("baseHeightTargetOffset",0)
            self.base_height_target = -lower_bound_z + base_height_tareget_offset
            self.cfg["env"]["baseInitState"]["pos"][2] = float(self.base_height_target+base_height_offset)
            print(f"{bc.WARNING}[infer from URDF] target_base_height = {self.base_height_target:.4f} {bc.ENDC}")
            print(f"{bc.WARNING}[infer from URDF] self.cfg['env']['baseInitState']['pos'][2] = {self.cfg['env']['baseInitState']['pos'][2]:.4f} {bc.ENDC}")


        # needed for foot height reward
        base_pos = self.asset_urdf.get_transform(self.base_name,collision_geometry=True)[:3,3]
        base_pos = torch.tensor(base_pos, dtype=torch.float, device=self.device)
        foot_pos_rel = np.stack([self.asset_urdf.get_transform(foot_name,collision_geometry=True)[:3,3] for foot_name in self.foot_names])
        self.default_foot_pos_rel = torch.tensor(foot_pos_rel, dtype=torch.float, device=self.device).unsqueeze(0) #(1,num_foot,3)
        foot_z_pos = np.mean(foot_pos_rel[:,2]) # z position of foot
        self.foot_height_offset: float = lower_bound_z - foot_z_pos
        print(f"{bc.WARNING}[infer from URDF] foot_height_offset = {self.foot_height_offset:.4f} {bc.ENDC}")

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        np.testing.assert_almost_equal(np.square(rot).sum(), 1, decimal=6, err_msg="env.baseInitState.rot should be normalized to 1")
        self.base_init_state = torch.tensor(pos + rot + v_lin + v_ang, dtype=torch.float, device=self.device)

        # time related
        self.decimation: int = self.cfg["env"]["control"]["decimation"]
        self.dt: float = self.cfg["sim"]["dt"]
        self.dt_inv = 1.0 / self.dt
        self.rl_dt = self.dt*self.decimation
        self.rl_dt_inv = 1.0 / self.rl_dt
        self.max_episode_length_s: float = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.rl_dt + 0.5)

        # other
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        
        self.enable_udp: bool = self.cfg["env"]["dataPublisher"]["enable"]
        if self.enable_udp:  # plotJuggler related
            self.data_publisher = DataPublisher(**self.cfg["env"]["dataPublisher"])
            self.items_to_publish = self.cfg["env"]["dataPublisher"].get("keys", None)
            self.data_root_label = self.cfg["env"]["data_root_label"]
            self.sim2real_publisher = DataPublisher(target_url="udp://localhost:9876",enable=True)

        gravity_norm = np.linalg.norm(self.cfg["sim"]["gravity"])

        # reward scales
        cfg_rew: Dict[str, Any] = self.cfg["env"]["learn"]["reward"]

        # self.rew_group: list = self.cfg["env"]["learn"].get("reward_group",[])
        # rew_names = set(cfg_rew.keys())
        # for group in self.rew_group:
        #     rew_names-=set(group)
        # self.rew_group.append(list(rew_names))


        self.rew_scales = {key: rew_item["scale"] for key,rew_item in cfg_rew.items()}
        for key in self.rew_scales:
            self.rew_scales[key] = float(self.rew_scales[key]) * self.rl_dt
        # do not scale termination reward
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]

        # self.torque_penalty_bound = self.cfg["env"]["learn"].get("torquePenaltyBound", 0.0)
        # print(f"torque penalty bound = {self.torque_penalty_bound}")

        # self.rew_lin_vel_xy_exp_scale: float = cfg_rew["lin_vel_xy"]["exp_scale"]
        # self.rew_lin_vel_z_exp_scale: float = cfg_rew["lin_vel_z"]["exp_scale"]
        # self.rew_ang_vel_z_exp_scale: float = cfg_rew["ang_vel_z"]["exp_scale"]

        # self.rew_orient_exp_scale: float = cfg_rew["orientation"]["exp_scale"]
        self.rew_foot_orient_exp_scale: float = cfg_rew["foot_orientation"]["exp_scale"]

        self.rew_action_rate_exp_scale: float = cfg_rew["action_rate"]["exp_scale"]
        
        self.rew_dof_acc_exp_scale: float = cfg_rew["dof_acc"]["exp_scale"]

        # self.rew_dof_jerk_exp_scale: float = cfg_rew["dof_jerk"]["exp_scale"]

        # self.rew_impact_exp_scale: float = cfg_rew["impact"]["exp_scale"]

        self.rew_lin_vel_exp_scale: torch.Tensor =cfg_rew["lin_vel"]["exp_scale"] * \
            torch.tensor(cfg_rew["lin_vel"]["normalize_by"],dtype=torch.float, device=self.device)
        self.rew_ang_vel_exp_scale: torch.Tensor =cfg_rew["ang_vel"]["exp_scale"] * \
            torch.tensor(cfg_rew["ang_vel"]["normalize_by"],dtype=torch.float, device=self.device)
        
        self.rew_foot_forward_exp_scale: float = cfg_rew["foot_forward"]["exp_scale"]

        self.rew_foot_pos_exp_scale: torch.Tensor = cfg_rew["foot_pos"]["exp_scale"] * \
            torch.tensor(cfg_rew["foot_pos"]["normalize_by"],dtype=torch.float, device=self.device)
        
        self.foot_contact_threshold: float = self.cfg["env"]["learn"]["foot_contact_threshold"]
        self.max_foot_contact_force = 100  # [N] # todo refactor

        # foot_height reward
        self.foot_height_clamp_max = cfg_rew["foot_height"]["clamp_max"]

        self.rew_dof_force_target_exp_scale: float = cfg_rew["dof_force_target"]["exp_scale"]
        dof_force_limit = torch.tensor(self.dof_props["effort"],dtype=torch.float, device=self.device)[self.actuated_dof_mask]
        assert torch.all(dof_force_limit < 10000), f"dof_force_limit={dof_force_limit} double check this value is correct"

        self.actuated_dof_force_max = dof_force_limit
        self.actuated_dof_force_min = -dof_force_limit
        # self.rew_dof_force_target_exp_scale= (self.rew_dof_force_target_exp_scale/ (dof_force_limit*self.num_actuated_dof))


        # # base height reward: reverse bell shaped curve
        # # https://researchhubs.com/post/maths/fundamentals/bell-shaped-function.html
        # a, b = self.cfg["env"]["learn"].get("baseHeightRewardParams", [0.04, 3])
        # self.base_height_rew_a, self.base_height_rew_b = float(a), float(b)

        self.rew_base_height_exp_scale = cfg_rew["base_height"]["exp_scale"]

        # min air time and stance time in seconds
        self.air_time_offset = float(cfg_rew["air_time"]["offset"])
        self.stance_time_offset = float(cfg_rew["stance_time"]["offset"])

        # ramdomize:push robot
        randomize = self.cfg["env"]["randomize"]
        self.should_push_robots = randomize["push"]["enable"]
        self.push_interval = int(randomize["push"]["interval_s"] / self.rl_dt + 0.5)
        self.push_vel_min = torch.tensor(randomize["push"]["velMin"], dtype=torch.float, device=self.device)
        self.push_vel_max = torch.tensor(randomize["push"]["velMax"], dtype=torch.float, device=self.device)

        # randomize: init_dof_pos
        self.randomize_init_dof_pos = randomize["initDofPos"]["enable"]
        self.randomize_init_dof_pos_range = randomize["initDofPos"]["range"]

        # randomize: init_dof_vel
        self.randomize_init_dof_vel = randomize["initDofVel"]["enable"]
        self.randomize_init_dof_vel_range = randomize["initDofVel"]["range"]

        # randomize: erfi
        self.enable_erfi:bool = randomize["erfi"]["enable"]
        self.erfi_rfi_range:Iterable[float] = randomize["erfi"]["rfi_range"] # random force injection
        self.erfi_rao_range:Iterable[float] = randomize["erfi"]["rao_range"] # random actuation offset
        if self.enable_erfi: 
            self.erfi_rao = torch.empty(self.num_envs,self.num_actuated_dof, dtype=torch.float, device=self.device).uniform_(*self.erfi_rao_range)
            self.erfi_rfi = torch.empty(self.num_envs,self.num_actuated_dof, dtype=torch.float, device=self.device).uniform_(*self.erfi_rfi_range)
        else:
            self.erfi_rfi = 0

        # randomize: dof_strength
        self.randomize_dof_strength = randomize["dof_strength"]["enable"]
        if self.randomize_dof_strength:
            self.dof_strength_range:Iterable[float] = randomize["dof_strength"]["range"]
            self.dof_strength = torch.empty(self.num_envs,self.num_actuated_dof, dtype=torch.float, device=self.device).uniform_(*self.dof_strength_range)
        else:
            self.dof_strength = torch.ones(self.num_envs,self.num_actuated_dof, dtype=torch.float, device=self.device)

        # random force purturbation
        self.randomize_body_force = randomize["body_force"]["enable"]
        self.force_scale=0
        if self.randomize_body_force:
            self.force_scale = randomize["body_force"]["scale"]
            self.force_log_prob_range = np.log(randomize["body_force"]["prob_range"])
            force_decay_time_constant:float = randomize["body_force"]["decay_time_constant"]
            self.force_decay = np.exp(-self.rl_dt/force_decay_time_constant)
            self.random_force_prob = torch.empty(self.num_envs, device=self.device).uniform_(*self.force_log_prob_range).exp_()
        # object apply random forces parameters
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # randomize default dof pos
        # possibly check out https://pytorch.org/docs/stable/generated/torch.randn.html
        self.randomize_default_dof_pos = randomize["default_dof_pos"]["enable"]
        if self.randomize_default_dof_pos:
            self.default_dof_pos_range = randomize["default_dof_pos"]["range"]
            self.default_dof_pos+=torch.empty_like(self.default_dof_pos).uniform_(*self.default_dof_pos_range)

        # randomize action delay
        self.randomize_action_delay = randomize["action_delay"]["enable"]
        if self.randomize_action_delay:
            self.action_delay_log_range = np.log(randomize["action_delay"]["range"])
            self.action_delay = torch.empty((self.num_envs,1), dtype=torch.float, device=self.device).uniform_(*self.action_delay_log_range).exp_()
            
        self.randomize_projected_gravity_delay = randomize["projected_gravity_delay"]["enable"]
        if self.randomize_projected_gravity_delay:
            self.projected_gravity_delay_log_range = np.log(randomize["projected_gravity_delay"]["range"])
            self.projected_gravity_delay = torch.empty((self.num_envs,1), dtype=torch.float, device=self.device).uniform_(*self.projected_gravity_delay_log_range).exp_()

        # heightmap
        self.init_height_points()

        self.is_train = False
        if  self.cfg["test"] in ["train", False]:
            self.is_train = True

        # passive dynamics
        self.enable_passive_dynamics = self.cfg["env"]["learn"].get("enablePassiveDynamics", False)
        self.passive_curriculum = self.cfg["env"]["learn"].get("passiveCurriculum", False)
        self.passive_curriculum = self.passive_curriculum and self.is_train # only train with curriculum

        self.action_is_on_rate = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)

        infer_action = self.cfg["env"]["numActions"] == "infer"
        if infer_action:
            self.cfg["env"]["numActions"] = len(self.actuated_dof_names)
            if self.enable_passive_dynamics:
                self.min_action_is_on:float = self.cfg["env"]["learn"]["action_is_on_min"] # 0.1  
                self.action_is_on_sigmoid_k:float = self.cfg["env"]["learn"]["action_is_on_sigmoid_k"]
                self.cfg["env"]["numActions"] = len(self.actuated_dof_names)*2
                self.duration_since_action_switch = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)
                self.last_action_is_on = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)
            
        # observation dimensions of specific items
        self.obs_dim_dict = {
            "linearVelocity": 3,
            "angularVelocity": 3,
            "projectedGravity": 3,
            "projected_gravity_xy": 2,
            "projected_gravity_filtered": 3,
            "commands": 3,  # vel_x,vel_y, vel_yaw, (excluding heading)
            "dofPosition": self.num_actuated_dof,
            "dofVelocity": self.num_actuated_dof,
            "dof_force_target": self.num_actuated_dof,
            "dof_strength": self.num_actuated_dof,
            "dofForce": self.num_actuated_dof,
            "heightMap": self.num_height_points,  # excluding the base origin measuring point
            "base_height": 1,
            "actions": self.cfg["env"]["numActions"],
            "last_actions": self.cfg["env"]["numActions"],
            "contact": self.num_foot,  # foot contact indicator
            "phase": self.num_foot*2, # phase of each foot (contact sequece)
            "contactTarget": self.num_foot,  # foot contact indicator
        }
        self.obs_names = tuple(self.cfg["env"]["observationNames"])

        

        self.num_stacked_obs_frame: int = cfg["env"].get("num_stacked_obs_frame", 1)
        self.num_stacked_state_frame: int = cfg["env"].get("num_stacked_state_frame", 1)

        # single frame observation
        self.num_obs_single_frame: int = np.sum(itemgetter(*self.obs_names)(self.obs_dim_dict))
        # stack multiple frames observation
        self.num_observations = self.num_obs_single_frame * self.num_stacked_obs_frame
        self.cfg["env"]["numObservations"] = self.num_observations
        print(f"\033[93m[inferring] numObservations={self.num_obs}\033[0m")
        # if self.cfg["env"]["numObservations"] == "infer":
        # self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        # self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        # assert self.cfg["env"]["numObservations"] == num_obs
        # self.num_observations = self.cfg["env"]["numObservations"]
        
        # buffer
        # delayed observations
        self.max_observation_delay_steps: int = cfg["env"].get("max_observation_delay_steps",0 )
        self.batched_obs_buf = buffer.BatchedRingTensorBuffer(
            buffer_len=self.max_observation_delay_steps+self.num_stacked_obs_frame,
            batch_size=self.num_envs, shape=self.num_obs_single_frame,dtype=torch.float, device=self.device)
            
        self.asymmetric_obs = self.cfg["env"].get("asymmetric_observations", False)
        if self.asymmetric_obs:
            self.state_names = tuple(self.cfg["env"]["stateNames"])
            self.num_state_single_frame: int = np.sum(itemgetter(*self.state_names)(self.obs_dim_dict))
            self.cfg["env"]["numStates"] = self.num_states = self.num_state_single_frame * self.num_stacked_state_frame
            self.batched_states_buf = buffer.BatchedRingTensorBuffer(
                buffer_len=self.num_stacked_state_frame,batch_size=self.num_envs, shape=self.num_state_single_frame,dtype=torch.float, device=self.device)
            print(f"\033[93m[inferring] numStates={self.cfg['env']['numStates']}\033[0m")
        self.num_states = self.cfg["env"].get("numStates", 0)

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self.num_actions = self.cfg["env"]["numActions"]
        self.control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = self.cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = self.cfg["env"].get("clipActions", np.Inf)

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames: int = 0

        # number of control steps
        self.control_steps: int = 0

        self.render_fps: int = self.cfg["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        self.record_frames_dir = os.path.join("recorded_frames", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
        # randomization_related_parameters
        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        # self.create_sim()
        
        # create plane/triangle mesh/heigh field
        self.terrain_type = self.cfg["env"]["terrain"]["terrainType"]
        self.terrain = Terrain(
            self.cfg["env"]["terrain"], num_robots=self.num_envs, device=self.device, gym=self.gym, sim=self.sim
        )
        if self.terrain_type in {'trimesh', 'heightfield'}:
            self.custom_origins = True
        elif self.terrain_type == 'plane':
            self.custom_origins = False
        else:
            raise NotImplementedError(f'Unsupported terrain type: {self.terrain_type}')
        
        # # asset force sensors
        # sensor_pose = gymapi.Transform()
        # sensor_options = gymapi.ForceSensorProperties()
        # sensor_options.enable_forward_dynamics_forces = False # for example gravity
        # sensor_options.enable_constraint_solver_forces = True # for example contacts
        # sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        # for index in self.foot_ids:
        #     self.gym.create_asset_force_sensor(self.asset, index, sensor_pose, sensor_options)
            
        self._create_envs()

        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.allocate_buffers()

        self.obs_dict = {}
        
        #######
        # get gym GPU state tensors
        self.dof_state_raw = self.gym.acquire_dof_state_tensor(self.sim)
        self.root_state_raw = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.net_contact_force_raw = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.rb_state_raw = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.dof_force_tensor_raw = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        # root_state: (num_actors, 13). 
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13]).
        self.root_state: torch.Tensor = gymtorch.wrap_tensor(self.root_state_raw)
        self.base_quat: torch.Tensor = self.root_state[:, 3:7] # [x,y,z,w]

        self.dof_state: torch.Tensor = gymtorch.wrap_tensor(self.dof_state_raw)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_acc = torch.zeros_like(self.dof_vel)

        # contact_force: (num_envs, num_bodies, xyz axis)
        self.contact_force: torch.Tensor = gymtorch.wrap_tensor(self.net_contact_force_raw).view(self.num_envs, -1, 3)

        # rb_state: (num_envs,num_rigid_bodies,13)
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13])
        self.rb_state: torch.Tensor = gymtorch.wrap_tensor(self.rb_state_raw).view(self.num_envs, -1, 13)

        # dof force tensor
        self.dof_force: torch.Tensor = gymtorch.wrap_tensor(self.dof_force_tensor_raw).view(self.num_envs, self.num_dof)

        # user-specified actuation forces (including passive joint, although we do not use it) [num_envs, num_dof]
        self.dof_actuation_force = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=torch.float)
        self.dof_actuation_force_tensor = gymtorch.unwrap_tensor(self.dof_actuation_force)


        # # force sensors
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        # self.sensor_forces = force_sensor_readings.view(self.num_envs, self.num_foot, 6)[..., :3]

        # reward episode sums (unscaled)
        def torch_zeros(): return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_sums = {key: torch_zeros() for key in self.rew_scales.keys()}

        # initialize some data used later on
        self.common_step_counter: int = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.noise_vec = torch.zeros((self.num_envs, self.num_obs_single_frame),dtype=torch.float, device=self.device)

        # self.max_action_steps = cfg["env"].get("max_action_steps",4)
        # self.batched_action_buf = BatchedRingTensorBuffer(buffer_len=self.max_action_steps, batch_size=self.num_envs, shape=self.num_actions, dtype=torch.float, device=self.device)

        # for force perturbation
        self.rb_force_mags =  gravity_norm * self.force_scale * self.actor_rigid_body_masses[:,:,None]

        # gravity_vec = [0,0,-1]
        gravity_vec = torch.tensor(get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device)
        self.base_gravity_vec = gravity_vec.repeat((self.num_envs, 1))
        self.projected_gravity = self.base_gravity_vec.clone()
        self.projected_gravity_filtered = self.base_gravity_vec.clone()
        self.foot_gravity_vec = gravity_vec.repeat((self.num_foot*self.num_envs, 1))

        # forward_vec = [1,0,0]
        forward_vec = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
        self.base_forward_vec_local = forward_vec.repeat((self.num_envs, 1))
        self.foot_forward_vec_local = forward_vec.repeat((self.num_foot*self.num_envs, 1))

        # idealized actuated dof force target
        self.actuated_dof_force_target = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)
        # after applying randomization and noise
        self.actuated_dof_force_target_actual = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device)

        self.action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.action_to_use = torch.zeros_like(self.action)
        self.last_action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.action_filt = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        # self.dof_pos_filt = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        # foot air time and stance time
        self.air_time = torch.zeros(self.num_envs, self.num_foot, dtype=torch.float, device=self.device)
        self.stance_time = torch.zeros(self.num_envs, self.num_foot, dtype=torch.float, device=self.device)
        self.last_foot_contact_force = torch.zeros(self.num_envs, self.num_foot, 3,
                                                   dtype=torch.float, device=self.device)
        # compute cost of transport metrics
        if self.enable_udp:
            buffer_len=160
            self.buffer_com_pos = buffer.BatchedRingTensorBuffer(buffer_len=buffer_len, batch_size=self.num_envs, shape=3, dtype=torch.float, device=self.device)
            self.buffer_dof_pow = buffer.BatchedRingTensorBuffer(buffer_len=buffer_len, batch_size=self.num_envs, shape=1, dtype=torch.float, device=self.device)
        
        # total mass of the actors [num_envs]
        total_mass = torch.tensor([np.sum([p.mass for p in body_props]) for body_props in self.actor_rigid_body_properties],dtype=torch.float, device=self.device)
        self.total_gravity = total_mass*gravity_norm
        self.total_gravity_inv = 1.0/self.total_gravity  # inverse of total gravity

        # single contact reward parameters
        self.max_single_contact: int = cfg_rew["single_contact"]["max_single_contact"]
        self.foot_multi_contact_grace_period: float = cfg_rew["single_contact"]["grace_period"]
        self.foot_multi_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.last_foot_contact = torch.zeros(self.num_envs, self.num_foot, dtype=torch.bool, device=self.device)

        cfg_guided_contact = self.cfg["env"]["learn"].get("guided_contact", {})
        self.guided_contact = cfg_guided_contact.get("enable", False)
        self.phase=None
        self.phase_sin_cos=None
        self.contact_target=None
        self.last_contact_target=None
        if self.guided_contact:
            self.phase_start_with_swing: bool = self.cfg["env"]["learn"]["guided_contact"]["phase_start_with_swing"]
            # normalized phase [0,1]
            self.episodic_phase_offset = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device)
            self.episodic_phase_offset[::2]= 0.5 # half of the episode is 0.5 phase offset
            self.phase_offset = torch.tensor(cfg_guided_contact["phase_offset"],dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            self.phase_freq: float = cfg_guided_contact["phase_freq"] # [Hz]
            self.phase_stance_ratio: float = cfg_guided_contact["phase_stance_ratio"] # [1]
            self.phase_swing_ratio = 1 - self.phase_stance_ratio
            self.last_contact_target = torch.ones(self.num_envs,self.num_foot, dtype=torch.bool, device=self.device)
            self.update_phase()

            # assume swing start from 0->self.phase_swing_ratio
            # compute 4th order polynomial coefficients
            sw = self.phase_swing_ratio
            q = 0.5 * sw
            A = np.array([
                [sw**2,sw**3,sw**4],
                [2*sw,3*sw**2,4*sw**3],
                [q**2,q**3,q**4]
            ])
            b = np.array([0,0,self.foot_height_clamp_max])
            self.foot_height_coeff = a =  np.linalg.solve(A, b) # y = a[0]*t**2+a[1]*t**3+a[2]*t**4
            self.foot_height_vel_coeff = np.array([2*a[0],3*a[1],4*a[2]])

            # t = np.linspace(0,sw,100)
            # y = a[0]*t**2+a[1]*t**3+a[2]*t**4
            # y_dot = 2*a[0]*t+3*a[1]*t**2+4*a[2]*t**3
            # y_dot_dot = 2*a[0]+6*a[1]*t+12*a[2]*t**2
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(3, 1, sharex=True)
            # ax[0].plot(t, y)
            # ax[0].set_ylabel('y')
            # ax[0].grid()
            # ax[1].plot(t, y_dot)
            # ax[1].set_ylabel('y_dot')
            # ax[1].grid()
            # ax[2].plot(t, y_dot_dot)
            # ax[2].set_ylabel('y_dot_dot')
            # ax[2].grid()
            # plt.tight_layout()
            # plt.show()

            # # get leg joint ids with respect to the each foot
            # graph = urdf_to_graph(self.asset_urdf)
            # leg_joint_names = [trace_edges(graph, node) for node in self.foot_names]
            # # # more general works for legs that has different DOFs
            # # self.leg_joint_ids =[torch.tensor([self.dof_dict[edge] for edge in edges],dtype=torch.int32, device=self.device) for edges in leg_joint_names]
            # self.leg_joint_ids = [[self.dof_dict[edge] for edge in edges] for edges in leg_joint_names]

            graph = urdf_to_graph(self.asset_urdf)
            # get mapping from a foot to the all the joints traced from that foot shaped [num_foot, num_dof] (including all passive and actuated joints)
            self.leg_to_dof_mask = torch.zeros(self.num_foot,self.num_dof, dtype=torch.bool, device=self.device)
            for k, foot_name in enumerate(self.foot_names):
                joint_traced_from_foot = trace_edges(graph, start_node=foot_name)
                joint_ids_traced_from_foot = [self.dof_dict[edge] for edge in joint_traced_from_foot]
                self.leg_to_dof_mask[k,joint_ids_traced_from_foot]=True
            self.leg_to_dof_mask = self.leg_to_dof_mask.repeat(self.num_envs,1,1) # expanded to [num_envs, num_foot, num_dof]

        if self.num_actuated_dof != self.num_dof: # has passive DOF
            self.pre_physics_step = self.pre_physics_step_with_passive_dof

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True
        
        # rendering
        self.set_viewer()

    # def reset_phase(self,env_ids):
    #     """Resets phase for specified environments."""
    #     self.last_contact_target[env_ids] = self.contact_target[env_ids]

    def update_phase(self):
        """update normalized contact phase for each foot, 
           stance phase: 0<phase<stance_ratio
           swing phase:  stance_ratio<phase<1
        """
        self.phase = (self.phase_freq * self.rl_dt * self.progress_buf.unsqueeze(1) +self.phase_offset + self.episodic_phase_offset) % 1
        self.phase_sin_cos = torch.column_stack((torch.sin(2*torch.pi*self.phase), torch.cos(2*torch.pi*self.phase)))
           
        if self.phase_start_with_swing:
            # NOTE! BREAKING CHANGE 09/07 : phase start with swing first instead
            self.contact_target = self.phase > self.phase_swing_ratio
        else:
            self.contact_target = self.phase <= self.phase_stance_ratio # 1 = stance, 0 = swing

        if self.keep_still_at_zero_command:
            # todo fix 0.5 bug
            should_stop = self.is_zero_command & (self.last_contact_target.sum(dim=-1)==self.num_foot) # & (self.projected_gravity[:,2]<-0.99) # 10 deg tilt        
            self.contact_target[should_stop] = 1 # set contact_target to 1 if zero command
        self.last_contact_target[:] = self.contact_target


    def set_viewer(self):
        """set viewers and camera events"""

        # rendering: virtual display
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            SCREEN_CAPTURE_RESOLUTION = (1027, 768)
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()

        self.enable_keyboard_operator: bool = self.cfg["env"]["viewer"]["keyboardOperator"]

        # todo: read from config
        if self.headless:
            self.viewer = None
            return

        # if running with a viewer, set up keyboard shortcuts and camera
            
        # self.enable_viewer_sync = False  # by default freeze the viewer until "V" is pressed
        self.enable_viewer_sync: bool = self.cfg["env"]["viewer"]["sync"]
    
        # subscribe to keyboard shortcuts
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        def subscribe_keyboard_event(key, event_str):
            self.gym.subscribe_viewer_keyboard_event(self.viewer, key, event_str)
        
        subscribe_keyboard_event(gymapi.KEY_ESCAPE, "QUIT")
        subscribe_keyboard_event(gymapi.KEY_V, "toggle_viewer_sync")
        subscribe_keyboard_event(gymapi.KEY_R, "record_frames")
        subscribe_keyboard_event(gymapi.KEY_9, "reset")

        if self.enable_keyboard_operator:
            subscribe_keyboard_event(gymapi.KEY_I, "vx+")
            subscribe_keyboard_event(gymapi.KEY_K, "vx-")
            subscribe_keyboard_event(gymapi.KEY_J, "vy+")
            subscribe_keyboard_event(gymapi.KEY_L, "vy-")
            subscribe_keyboard_event(gymapi.KEY_U, "heading+")
            subscribe_keyboard_event(gymapi.KEY_O, "heading-")
            subscribe_keyboard_event(gymapi.KEY_0, "v=0")
            subscribe_keyboard_event(gymapi.KEY_Y, "m+")
            subscribe_keyboard_event(gymapi.KEY_H, "m-")
            subscribe_keyboard_event(gymapi.KEY_P, "push")

            self.keyboard_operator_cmd = torch.zeros(3, dtype=torch.float, device=self.device)
            self.data_receiver = DataReceiver(port=9871,decoding="msgpack",broadcast=True)
            self.data_receiver_data_id = self.data_receiver.data_id # for check if data is new
            self.data_receiver.receive_continuously()
        
        subscribe_keyboard_event(gymapi.KEY_F, "toggle_viewer_follow")
        # switch camera follow target
        subscribe_keyboard_event(gymapi.KEY_LEFT_BRACKET, "ref_env-") # [
        subscribe_keyboard_event(gymapi.KEY_RIGHT_BRACKET, "ref_env+") # ]
        
        # set the camera position based on up axis
        # self.sim_params = self.gym.get_sim_params(self.sim)
        if self.sim_params.up_axis == gymapi.UP_AXIS_Z:
            self.cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            self.cam_target_pos = gymapi.Vec3(10.0, 15.0, 0.0)
        else:
            self.cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            self.cam_target_pos = gymapi.Vec3(10.0, 0.0, 15.0)

        self.cam_pos = torch.tensor(self.cfg["env"]["viewer"]["pos"], dtype=torch.float, device=self.device)
        self.cam_target_pos = torch.tensor(self.cfg["env"]["viewer"]["lookat"], dtype=torch.float, device=self.device)
        self.ref_env:int = int(self.cfg["env"]["viewer"]["refEnv"])%self.num_envs

        self.viewer_follow = self.cfg["env"]["viewer"]["follow"]
        self.viewer_follow_offset = torch.tensor(self.cfg["env"]["viewer"].get("follower_offset", [0.6, 1.2, 0.4]), dtype=torch.float, device=self.device)
        fs = self.rl_dt_inv
        filter_order=1
        from common.buffer import RingTensorFilterBuffer
        self.cam_pos_filter_buffer = RingTensorFilterBuffer(fs=fs, cut_off_frequency=1,filter_order=filter_order,shape=3,device=self.device)
        self.cam_pos_filter_buffer.fill(self.cam_pos)
        self.cam_target_pos_filter_buffer = RingTensorFilterBuffer(fs=fs, cut_off_frequency=1,filter_order=filter_order,shape=3,device=self.device)
        self.cam_target_pos_filter_buffer.fill(self.cam_target_pos)
        if self.viewer_follow:
            self.cam_target_pos = self.root_state[self.ref_env, :3].clone()
            self.cam_pos = self.viewer_follow_offset + self.cam_target_pos

        self.gym.viewer_camera_look_at(self.viewer, self.envs[self.ref_env], gymapi.Vec3(*self.cam_pos), gymapi.Vec3(*self.cam_target_pos))
        
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        if self.debug_viz:
            self.sphere_geom = gymutil.WireframeSphereGeometry(0.01, 5, 5, None, color=(1, 1, 0))
            self.sphere_geom_alt_color = gymutil.WireframeSphereGeometry(0.01, 5, 5, None, color=(1, 0, 0))


    def _parse_sim_params(self):
        """Parse the config dictionary for physics stepping settings.
        Returns
            IsaacGym SimParams object with updated settings.
        """
        config_sim = self.cfg["sim"]
        physics_engine = self.cfg["physics_engine"]
        
        sim_params = gymapi.SimParams()
        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        elif config_sim["up_axis"] == "y":
            sim_params.up_axis = gymapi.UP_AXIS_Y
        else:
            raise ValueError(f"Invalid physics up-axis: {config_sim['up_axis']}")

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        elif physics_engine == "flex":
            self.physics_engine = gymapi.SIM_FLEX
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])
        else:
            raise ValueError(f"Invalid physics engine backend: {self.cfg['physics_engine']}")

        return sim_params

    def _get_noise_scale_vec(self):
        """Calculates noise scaling factors for observations."""
        cfg_learn = self.cfg["env"]["learn"]
        self.add_noise = cfg_learn["addNoise"]
        noise_level = cfg_learn["noiseLevel"]
        noise_dict = {
            "linearVelocity": cfg_learn["linearVelocityNoise"] * noise_level * self.lin_vel_scale,
            "angularVelocity": cfg_learn["angularVelocityNoise"] * noise_level * self.ang_vel_scale,
            "projectedGravity": cfg_learn["gravityNoise"] * noise_level,
            "projected_gravity_filtered": cfg_learn["gravityNoise"] * noise_level,
            "projected_gravity_xy": cfg_learn["gravityNoise"] * noise_level,
            "commands": 0,
            "dofPosition": cfg_learn["dofPositionNoise"] * noise_level * self.dof_pos_scale,
            "dofVelocity": cfg_learn["dofVelocityNoise"] * noise_level * self.dof_vel_scale,
            "dof_force_target": 0,
            "dof_strength": 0,
            "dofForce": 0, # TODO, MAYBE ADD NOISE FOR DOF FORCE
            "heightMap": cfg_learn["heightMapNoise"] * noise_level * self.heightmap_scale,
            "base_height": 0,
            "actions": 0,  # previous actions
            "last_actions": 0,
            "contact": 0,  # foot contact
            "contactTarget":0,
            "phase": 0,
        }
        noise_vec_lists = [torch.ones(self.obs_dim_dict[name]) * noise_dict[name] for name in self.obs_names]
        noise_vec = torch.cat(noise_vec_lists, dim=-1).to(self.device)
        return noise_vec
    
    @property
    def asset_urdf(self):
        try:
            return self._asset_urdf
        except AttributeError:
            import yourdfpy
            self._asset_urdf = yourdfpy.URDF.load(
                self.asset_path,
                build_scene_graph=False,
                load_meshes=False,
                load_collision_meshes=True,
                build_collision_scene_graph=True
            )
            return self._asset_urdf

    def load_asset(self):
        """Loads the robot asset (URDF).
        Requires gym to be initialized
        """
        cfg_asset = self.cfg["env"]["urdfAsset"]
        if "root" not in cfg_asset:  # root directory
            cfg_asset["root"] = 'assets'  # relative to the legged_env folder
        if not os.path.isabs(cfg_asset["root"]):
            cfg_asset["root"] = os.path.abspath(
                os.path.join(os.path.dirname(to_absolute_path(__file__)), "./../../", cfg_asset["root"]))
        
        self.asset_path = os.path.join(cfg_asset["root"], cfg_asset["file"])

        # bitwise filter for elements in the same collisionGroup to mask off collision
        self.collision_filter = cfg_asset["collision_filter"]

        asset_options = gymapi.AssetOptions()
        # defaults
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        # asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_VELOCITY)

        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        # asset_options.override_inertia = True
        for attribute in cfg_asset["AssetOptions"]:
            if attribute == "vhacd_params":
                vhacd_params = cfg_asset["AssetOptions"]["vhacd_params"]
                for key in vhacd_params:
                    setattr(asset_options.vhacd_params, key, vhacd_params[key])
            elif hasattr(asset_options, attribute):
                setattr(asset_options, attribute, cfg_asset["AssetOptions"][attribute])
            else:
                print(f"{bc.WARNING}{attribute} not in AssetOptions!{bc.ENDC}")

        self.asset = self.gym.load_asset(self.sim, cfg_asset["root"], cfg_asset["file"], asset_options)
        self.dof_names = self.gym.get_asset_dof_names(self.asset)
        # name to index mapping
        self.dof_dict = self.gym.get_asset_dof_dict(self.asset)
        # number of DOF
        self.num_dof = self.gym.get_asset_dof_count(self.asset)

        try:
            self.passive_dof_names = get_matching_str(source="PASSIVE", destination=self.dof_names, case_sensitive=True, comment="passive DOF")
        except KeyError:
            self.passive_dof_names = []
        self.actuated_dof_names = list(sorted([d for d in self.dof_names if d not in self.passive_dof_names]))
        self.num_actuated_dof = len(self.actuated_dof_names)
        actuated_dof_ids = torch.tensor([self.dof_dict[n] for n in self.actuated_dof_names], device=self.device, dtype=torch.long)
        self.actuated_dof_mask = torch.zeros(self.num_dof, dtype=torch.bool, device=self.device)
        self.actuated_dof_mask[actuated_dof_ids] = True


        # dof properties
        # lower: lower limit of DOF. in [radians] or [meters]
        # upper: upper limit of DOF. in [radians] or [meters]
        # velocity: Maximum velocity of DOF. in [radians/s] or [meters/s]
        # effort: Maximum effort of DOF. in [N] or [Nm].
        # stiffness: DOF stiffness.    
        # damping: DOF damping.    
        # friction: DOF friction coefficient, a generalized friction force is calculated as DOF force multiplied by friction.
        # armature: DOF armature, a value added to the diagonal of the joint-space inertia matrix. Physically, it corresponds to the rotating part of a motor - which increases the inertia of the joint, even when the rigid bodies connected by the joint can have very little inertia.
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        # asset dof properties override
        asset_dof_properties = self.cfg["env"].get("assetDofProperties", {})
        if asset_dof_properties is not None:
            for key, value in asset_dof_properties.items():
                self.dof_props[key][:] = np.asarray(value,dtype=np.float32)  # used in set_actor_dof_properties
                print(f"overwrite asset dof [{key}]: {value}")

        # dof limit
        self.dof_soft_limit_lower = torch.tensor(
            self.cfg["env"].get("dof_soft_limit",{}).get("lower",self.dof_props['lower']), dtype=torch.float, device=self.device)
        self.dof_soft_limit_upper = torch.tensor(
            self.cfg["env"].get("dof_soft_limit",{}).get("upper",self.dof_props['upper']),dtype=torch.float, device=self.device)

        dof_margin = self.cfg["env"]["learn"].get("dofLimitMargins", 0)
        dof_margin = torch.tensor(dof_margin, dtype=torch.float, device=self.device)
        self.dof_limit_lower = torch.tensor(self.dof_props['lower'], dtype=torch.float, device=self.device) + dof_margin
        self.dof_limit_upper = torch.tensor(self.dof_props['upper'], dtype=torch.float, device=self.device) - dof_margin
        # self.dof_limit_lower = self.dof_soft_limit_lower + dof_margin
        # self.dof_limit_upper = self.dof_soft_limit_upper - dof_margin

        # body
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self.body_names = self.gym.get_asset_rigid_body_names(self.asset)

        asset_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset)
        asset_rigid_body_id_dict = {value: key for key, value in asset_rigid_body_dict.items()}

        # body: base
        self.base_name = cfg_asset.get("baseName", None)
        if self.base_name is None:  # infer base_name
            self.base_name = self.asset_urdf.base_link
        else:
            self.base_name = get_matching_str(source=self.base_name, destination=self.body_names, comment="base_name")[0]
        self.base_id = asset_rigid_body_dict[self.base_name]
        
        # hip_name = cfg_asset.get("hipName",None)
        # hip_names = get_matching_str(source=hip_name, destination=self.body_names, comment="hip_name")
        # self.hip_ids = torch.tensor([asset_rigid_body_dict[n] for n in hip_names], 
        #                             dtype=torch.long, device=self.device)

        foot_name = cfg_asset.get("footName", None)
        if foot_name is None:  # infering the feet are leaf links, they do not appear in any joint.parent
            self.foot_names = get_leaf_nodes(urdf=self.asset_urdf, 
                collapse_fixed_joints=self.cfg["env"]["urdfAsset"]["AssetOptions"]["collapse_fixed_joints"])
        else:
            self.foot_names = get_matching_str(source=foot_name, destination=self.body_names, comment="foot_name")
        self.num_foot = len(self.foot_names)
        self.foot_ids = torch.tensor([asset_rigid_body_dict[n] for n in self.foot_names], 
                                     dtype=torch.long, device=self.device)
        assert(self.foot_ids.numel() > 0)

        # TODO CHANGE KNEE COLLISIONN TO A MORE GENARIC TYPE OF REWARD: collision for anything other than the foot maybe?
        knee_name = cfg_asset.get("kneeName", None)
        if knee_name is None:
            # if knee_name is None: exclude base link and feet, include all other links
            exclude_link_names = set(self.foot_names)
            exclude_link_names.add(self.base_name)
            knee_names = set(asset_rigid_body_dict.keys()) - exclude_link_names
        else:
            knee_names = get_matching_str(source=knee_name, destination=self.body_names, comment="knee_name")
        self.knee_ids = torch.tensor([asset_rigid_body_dict[n] for n in knee_names],
                                     dtype=torch.long, device=self.device)
        assert (self.knee_ids.numel() > 0)
        # # joints
        # hip_joint_name = cfg_asset["hipJointName"]
        # asset_dof_dict = self.gym.get_asset_dof_dict(self.asset)
        # asset_dof_id_dict = {value: key for key, value in asset_dof_dict.items()}
        # dof_hip_names = get_matching_str(source=hip_joint_name,destination=self.dof_names, comment="hip_joint_name")
        # self.dof_hip_ids = torch.tensor([asset_dof_dict[n] for n in dof_hip_names], dtype=torch.long, device=self.device)   

        print(f"base = {self.base_name}: {self.base_id}")
        # print(f"hip = {dict(zip(hip_names,self.hip_ids.tolist()))}")
        print(f"knee = {dict(zip(knee_names,self.knee_ids.tolist()))}")
        print(f"foot = {dict(zip(self.foot_names,self.foot_ids.tolist()))}")
        # print(f"dof_hip = {dict(zip(dof_hip_names,self.dof_hip_ids.tolist()))}")
        assert self.base_id != -1
        # assert len(self.dof_hip_ids)==4
        # assert self.dof_hip_ids.tolist() == [0, 3, 6, 9]
        ####################

        marker_pair_names = cfg_asset.get("marker_pair_names",[])
        self.num_marker_pairs = len(marker_pair_names)
        if len(marker_pair_names) > 0:
            self.marker_pair_l0 = torch.tensor(cfg_asset["marker_pair_length"], dtype=torch.float, device=self.device)
            self.last_marker_pair_length = self.marker_pair_l0.repeat(self.num_envs,1)

            self.marker_pair_names = [get_matching_str(
                source=marker_pair_name, destination=self.body_names, case_sensitive=True,comment="marker_pairs") 
                for marker_pair_name in marker_pair_names]
            print("marker pair names:", self.marker_pair_names)
            self.marker_pair_ids = torch.tensor(
                [[asset_rigid_body_dict[n] for n in marker_pair] for marker_pair in self.marker_pair_names],
                dtype=torch.long, device=self.device)
             
        
    def _create_envs(self):
        """Creates multiple environments with randomized properties."""

        randomize = self.cfg["env"]["randomize"]
        # friction randomization
        randomize_friction:bool = randomize["friction"]["enable"]
        if randomize_friction:
            rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(self.asset)
            friction_buckets = torch.empty(self.num_envs, device=self.device,dtype=torch.float).uniform_(*randomize["friction"]["range"])
        # baseMass randomization
        randomize_base_mass:bool = randomize["baseMass"]["enable"]
        if randomize_base_mass:
            baseMass_buckets = torch.empty(self.num_envs, device=self.device,dtype=torch.float).uniform_(*randomize["baseMass"]["range"])
            # added_masses = np.random.uniform(*self.cfg["env"]["learn"]["addedMassRange"], self.num_envs)
        randomize_base_inertia_origin:bool = randomize["baseInertiaOrigin"]["enable"]
        if randomize_base_inertia_origin:
            origin_range = torch.tensor(randomize["baseInertiaOrigin"]["range"], device=self.device, dtype=torch.float)
            origin_buckets = torch_rand_tensor(lower=origin_range[:,0], upper=origin_range[:,1], shape=(self.num_envs, 3), device=self.device)

        # radomize link mass
        randomize_link_mass = randomize["link_mass"]["enable"]
        if randomize_link_mass:
            link_mass_range = randomize["link_mass"]["range"]
            link_mass_buckets = torch.empty(self.num_envs,self.num_bodies, device=self.device,dtype=torch.float).uniform_(*link_mass_range)
            link_inv_mass_buckets = 1.0 / link_mass_buckets
        # randomize link inertia
        randomize_link_inertia = randomize["link_inertia"]["enable"]
        if randomize_link_inertia:
            link_inertia_range = randomize["link_inertia"]["range"]
            link_inertia_buckets = torch.empty(self.num_envs,self.num_bodies, device=self.device,dtype=torch.float).uniform_(*link_inertia_range)
            # link_inertia_buckets = link_inertia_buckets.unsqueeze(-1).repeat_interleave(9, dim=-1)
            link_inv_inertia_buckets = 1.0 / link_inertia_buckets
            # self.link_mass+=torch.empty_like(self.link_mass).uniform_(*self.link_mass_range)

        # randomize base_pos_xy
        self.randomize_base_init_pos_xy = randomize["base_init_pos_xy"]["enable"]
        self.randomize_base_init_pos_xy_range = self.cfg["env"]["randomize"]["base_init_pos_xy"]["range"]
        if not self.randomize_base_init_pos_xy:
            self.randomize_base_init_pos_xy_range = [0,0]
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
        
        spacing = self.cfg["env"]['envSpacing']
        num_per_row = int(np.sqrt(self.num_envs))

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        if self.custom_origins:
            spacing = 0.0
            # self.randomize_base_init_pos_xy_range = (-1.0, 1.0)  # TODO refactor into config
            base_pos_xy_offset = torch.empty(self.num_envs, 2, dtype=torch.float,device=self.device).uniform_(*self.randomize_base_init_pos_xy_range)

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = []
        self.envs = []
        self.actor_rigid_body_properties = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain.env_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += base_pos_xy_offset[i]
                start_pose.p = gymapi.Vec3(*pos)
            actor_handle = self.gym.create_actor(env_handle, self.asset, start_pose, "actor", i, self.collision_filter, 0)
            
            # dof_force_tensor
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            actor_rigid_shape_prop = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            if randomize_friction:
                for s in range(len(actor_rigid_shape_prop)):
                    actor_rigid_shape_prop[s].friction = friction_buckets[i]
                self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, actor_rigid_shape_prop)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, self.dof_props)
            
            actor_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if randomize_base_mass:
                actor_rigid_body_props[self.base_id].mass += baseMass_buckets[i]
            if randomize_base_inertia_origin:
                actor_rigid_body_props[self.base_id].com += gymapi.Vec3(*origin_buckets[i])
            if randomize_link_mass:
                for b in range(self.num_bodies):
                    actor_rigid_body_props[b].mass*=link_mass_buckets[i, b]
                    actor_rigid_body_props[b].invMass*=link_inv_mass_buckets[i, b]
            if randomize_link_inertia:
                for b in range(self.num_bodies):
                    actor_rigid_body_props[b].inertia.x.x *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].inertia.x.y *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].inertia.x.z *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].inertia.y.x *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].inertia.y.y *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].inertia.y.z *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].inertia.z.x *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].inertia.z.y *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].inertia.z.z *= link_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.x.x *= link_inv_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.x.y *= link_inv_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.x.z *= link_inv_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.y.x *= link_inv_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.y.y *= link_inv_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.y.z *= link_inv_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.z.x *= link_inv_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.z.y *= link_inv_inertia_buckets[i, b]
                    actor_rigid_body_props[b].invInertia.z.z *= link_inv_inertia_buckets[i, b]            
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, actor_rigid_body_props, recomputeInertia=False)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.actor_rigid_body_properties.append(actor_rigid_body_props)
        
        self.actor_rigid_body_masses = torch.tensor(
            [[prop.mass for prop in props] for props in self.actor_rigid_body_properties], dtype=torch.float, device=self.device)

    def check_termination(self):
        """Checks if the episode should terminate."""
        # base tilt 45 degree or base contact with ground
        self.reset_buf = torch.logical_or(self.projected_gravity[:,2]>-0.7, square_sum(self.contact_force[:, self.base_id, :]) > 1.0)
        # self.reset_buf = square_sum(self.contact_force[:, self.base_id, :]) > 1.0

        if not self.allow_knee_contacts:
            knee_contact = square_sum(self.contact_force[:, self.knee_ids, :]) > 1.0
            self.reset_buf |= torch.any(knee_contact, dim=1)

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 0
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.timeout_buf = time_out & (self.reset_buf == 0)
        self.reset_buf[time_out] = 1

    def compute_observations(self):
        """Computes observations for the current state."""
        self.get_heights()
        heights = torch.clip(self.heights_relative - self.base_height_target, -1.0, 1.0) * self.heightmap_scale

        obs_dict = {
            "linearVelocity": self.base_lin_vel * self.lin_vel_scale,
            "angularVelocity": self.base_ang_vel * self.ang_vel_scale,
            "projectedGravity": self.projected_gravity,
            "projected_gravity_xy": self.projected_gravity[:,:2],
            "projected_gravity_filtered":self.projected_gravity_filtered,
            "commands": self.commands[:, :3] * self.commands_scale,
            "dofPosition": self.dof_pos[:,self.actuated_dof_mask] * self.dof_pos_scale,
            "dofVelocity": self.dof_vel[:,self.actuated_dof_mask] * self.dof_vel_scale,
            "dof_force_target": self.actuated_dof_force_target * self.dof_force_target_scale,
            "actuated_dof_force_target_actual": self.actuated_dof_force_target_actual * self.dof_force_target_scale,
            # "dofForce": self.dof_force[:,self.actuated_dof_mask] * self.dof_force_scale,
            "dof_strength": self.dof_strength,
            "base_height": heights[:, self.num_height_points+self.base_id],
            "heightMap": heights[:, :self.num_height_points],
            "last_actions": self.last_action,
            "actions": self.action,
            "contact": self.foot_contact,
            "contactTarget": self.contact_target,
            "phase": self.phase_sin_cos,
        }
        # update observation buffer
        obs_buf_single_frame = torch.cat(itemgetter(*self.obs_names)(obs_dict), dim=-1)
        if self.add_noise:
            self.noise_vec.uniform_(-1.0, 1.0).mul_(self.noise_scale_vec)  # scaled noise vector
            obs_buf_single_frame += self.noise_vec
        # delayed observations
        # self.batched_obs_buf.add_and_fill_batch(obs_buf_single_frame)
        self.batched_obs_buf.add(obs_buf_single_frame)
        # self.obs_buf = self.batched_obs_buf[self.max_observation_delay_steps].clone()
        self.obs_buf = torch.transpose(self.batched_obs_buf.get_latest_n(self.num_stacked_obs_frame,offset=self.max_observation_delay_steps).clone(),0,1).reshape(self.num_envs, self.num_obs)
        # self.obs_buf = self.batched_obs_buf[:self.num_stacked_obs_frame].clone().

        if self.asymmetric_obs: # update state buffer
            states_buf_single_frame = torch.cat(itemgetter(*self.state_names)(obs_dict), dim=-1)
            # self.batched_states_buf.add_and_fill_batch(states_buf)
            self.batched_states_buf.add(states_buf_single_frame)
            self.states_buf = torch.transpose(self.batched_states_buf.get_latest_n(self.num_stacked_state_frame).clone(),0,1).reshape(self.num_envs, self.num_states)
        


    def compute_reward(self):
        """Computes the reward for the current state and action."""
        rew = {}
        # velocity tracking reward

        desired_vel = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float)
        desired_vel[:, [0,1,5]] = self.commands[:, [0,1,2]]
        # linear velocity error xyz
        lin_vel_error: torch.Tensor = desired_vel[:, :3] - self.base_lin_vel
        ang_vel_error: torch.Tensor = desired_vel[:, 3:] - self.base_ang_vel

        rew["lin_vel"] = exp_weighted_square_sum(lin_vel_error, self.rew_lin_vel_exp_scale)
        rew["ang_vel"] = exp_weighted_square_sum(ang_vel_error, self.rew_ang_vel_exp_scale)

        # lin_vel_error = square_sum(self.commands[:, :2] - self.base_lin_vel[:, :2])
        # ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # rew["lin_vel_xy"] = torch.exp(self.rew_lin_vel_xy_exp_scale*lin_vel_error)
        # rew["ang_vel_z"] = torch.exp(self.rew_ang_vel_z_exp_scale*ang_vel_error)

        # # other base velocity penalties
        # rew["lin_vel_z"] = torch.square(self.base_lin_vel[:, 2])
        # rew["ang_vel_xy"] = square_sum(self.base_ang_vel[:, :2])

        # orientation penalty
        rew["orientation"] = square_sum_clamp_max(self.projected_gravity[:, :2], max=0.1)
        # rew["orientation"] = square_sum(self.projected_gravity[:, :2])
        # projected_gravity_xy = square_sum(self.projected_gravity[:, :2])
        # rew["orientation"]=torch.exp(self.rew_orient_exp_scale*projected_gravity_xy)
        # rew["orientation"]=exp_square_sum(self.projected_gravity[:, :2],exp_scale=self.rew_orient_exp_scale)
        # rew["orientation"]=exp_square(torch.arccos(-self.projected_gravity[:, 2]),exp_scale=self.rew_orient_exp_scale) # changed 8/20 # numerically unstable,dont use it

        # base height penalty
        # rew["base_height"] = torch.square(self.heights_relative[:, self.num_height_points] - self.target_base_height)

        # exponential reward
        # base_height_error = torch.clamp_max(self.base_height - self.base_height_target,0)
        base_height_error = self.base_height - self.base_height_target

        rew["base_height"] =torch.exp(torch.square(base_height_error)*self.rew_base_height_exp_scale)
        
        # # bell-shaped reward
        # rew["base_height"] = reverse_bell(
        #     self.base_height, a=self.base_height_rew_a, b=self.base_height_rew_b, c=self.target_base_height
        # )

        # torque penalty
        # rew["dof_force_target"] = torch.exp((self.actuated_dof_force_target.abs()*self.rew_dof_force_target_exp_scale).sum(dim=-1))
        rew["dof_force_target"] = torch.exp((self.actuated_dof_force_target.abs()*self.total_gravity_inv.view(self.num_envs,1)*self.rew_dof_force_target_exp_scale).sum(dim=-1)) # normalize by total gravity

        # rew["dof_force_target"] = out_of_float_bound_squared_sum(
        #     self.dof_force_target, -self.torque_penalty_bound, self.torque_penalty_bound
        # )
        # rew["dof_force_target"] = out_of_bound_norm(self.dof_force_target, -self.torque_penalty_bound, self.torque_penalty_bound)

        # joint acc penalty
        # rew["dof_acc"] = torch.square(self.last_dof_vel - self.dof_vel).sum(dim=1)

        rew["dof_acc"] = exp_square_sum(self.dof_acc, self.rew_dof_acc_exp_scale)

        # rew["dof_jerk"] = exp_square_sum(self.dof_jerk, self.rew_dof_jerk_exp_scale)
        
        # rew["dof_acc"] = torch.abs(dof_acc).sum(dim=1)  #TODO Check this

        # joint vel penalty
        rew["dof_vel"] = square_sum(self.dof_vel)

        actuated_dof_pos_delta = (self.dof_pos - self.desired_dof_pos)[:,self.actuated_dof_mask].abs()

        if self.keep_still_at_zero_command:
            # joint position penalty
            rew["dof_pos"] = actuated_dof_pos_delta.sum(dim=1)
            rew["dof_pos"][~self.is_zero_command] = 0 # do not count if it is not zero command

        # rew["dof_pos"] = (self.dof_pos_filt - self.desired_dof_pos).abs().sum(dim=1)
        
        # joint power penalty
        # rew["dof_pow"] = (self.dof_vel[:,self.actuated_dof_mask] * self.actuated_dof_force_target).abs().sum(dim=1)
        # power to weight ratio
        rew["dof_pow"] = (self.dof_vel[:,self.actuated_dof_mask] * self.actuated_dof_force_target).abs().sum(dim=1)*self.total_gravity_inv

        # penalty for position exceeding dof limit
        rew["dof_limit"] = out_of_bound_square_sum(self.dof_pos, self.dof_limit_lower, self.dof_limit_upper)

        
        # collision penalty
        knee_collision = square_sum(self.contact_force[:, self.knee_ids, :], dim=2) > 1.0
        rew["collision"] = torch.sum(knee_collision, dim=1, dtype=torch.float)  # sum vs any ?

        # foot impact penalty (num_envs,num_foot,3)
        rew["impact"] = torch.clamp(self.foot_contact_force[:, :, 2]*self.total_gravity_inv.view(self.num_envs,1)-1,min=0,max=2).square().sum(dim=1)
        # rew["impact"] = exp_square_sum(foot_contact_diff, self.rew_impact_exp_scale)

        # rew["impact"] = foot_contact_diff.view(self.num_envs,-1).abs().sum(dim=1)
        # foot_contact_diff = self.foot_contact_force - self.last_foot_contact_force
        # rew["impact"] = torch.norm(foot_contact_diff,dim=2).sum(dim=1)

        # stumbling penalty
        stumble = (torch.norm(self.contact_force[:, self.foot_ids, :2], dim=2) > 5.0) * (
            torch.abs(self.contact_force[:, self.foot_ids, 2]) < 1.0
        )
        rew["stumble"] = torch.sum(stumble, dim=1, dtype=torch.float)

        # foot slip penalty
        rew["slip"] = (self.foot_lin_vel.square().sum(dim=2) * self.foot_contact_filt).sum(dim=1)

        # action penalty
        rew["action"] = self.action.abs().sum(dim=1)
        # rew["action"] = torch.square(self.actions).sum(dim=1)
        # rew["action"] = self.actions_filt.abs().sum(dim=1)  # filtered

        # action rate penalty
        self.action_rate = (self.action - self.last_action) * self.rl_dt_inv
        rew["action_rate"] = exp_square_sum(self.action_rate[:,:self.num_actuated_dof], self.rew_action_rate_exp_scale)

        # self.is_zero_command = torch.norm(self.commands[:, :2], dim=1) < self.command_zero_threshold
        # nonzero_command = torch.norm(self.commands[:, :2], dim=1) >= self.command_zero_threshold
        foot_no_contact = ~self.foot_contact_filt

        # air time reward (reward long swing)
        first_contact = (self.air_time > 0.0) * self.foot_contact_filt
        self.air_time += self.rl_dt
        # reward only on first contact with the ground
        # rew["air_time"] = torch.sum((self.air_time + self.air_time_offset)*first_contact, dim=1, dtype=torch.float)
        rew["air_time"] = torch.sum((self.air_time + self.air_time_offset)*first_contact, dim=1, dtype=torch.float).clamp_max_(0) # no reward for going beyond


        self.air_time *= foot_no_contact  # reset if contact

        # foot stance time reward (reward long stance)
        first_no_contact = (self.stance_time > 0.0) * foot_no_contact
        self.stance_time += self.rl_dt
        # reward only on first leaving the ground
        # rew["stance_time"] = torch.sum((self.stance_time + self.stance_time_offset) * first_no_contact, dim=1, dtype=torch.float)
        rew["stance_time"] = torch.sum((self.stance_time + self.stance_time_offset) * first_no_contact, dim=1, dtype=torch.float).clamp_max_(0) # no reward for going beyond


        self.stance_time *= self.foot_contact_filt  # reset if no contact

        # # reward contact at zero command
        # rew["contact"] = torch.sum(self.foot_contact_filt, dim=1,dtype=torch.float)* (~nonzero_command)

        # penalize high contact forces
        contact_force_norm = torch.norm(self.contact_force[:, self.foot_ids, :], dim=-1)
        rew["contact_force"] = torch.sum((contact_force_norm - self.max_foot_contact_force).clip(min=0.0), dim=1)


        foot_num_contact = torch.sum(self.foot_contact_filt, dim=1, dtype=torch.int) # TODO
        foot_multi_contact = torch.logical_not(foot_num_contact == self.max_single_contact)
        self.foot_multi_contact_time[foot_multi_contact]+=self.rl_dt
        self.foot_multi_contact_time[~foot_multi_contact] = 0.0
        
        rew["single_contact"] = torch.logical_or(
            self.foot_multi_contact_time <self.foot_multi_contact_grace_period, 
            self.is_zero_command).type(torch.float)

        foot_projected_gravity_xy = square_sum(self.foot_projected_gravity[...,:2]).mean(dim=-1)
        rew["foot_orientation"]=torch.exp(self.rew_foot_orient_exp_scale*foot_projected_gravity_xy)
        
        foot_pos_rel_error = torch.abs(self.default_foot_pos_rel - self.foot_pos_rel_yaw).mean(dim=1)
        rew["foot_pos"] = exp_weighted_square_sum(foot_pos_rel_error, self.rew_foot_pos_exp_scale,dim=-1)

        # self.foot_forward: self.num_envs, self.num_foot, 3
        base_forward_xy = normalize(self.base_forward[...,:2])
        foot_forward_xy = normalize(self.foot_forward[...,:2])
        
        rew["foot_forward"] = torch.exp(self.rew_foot_forward_exp_scale*(1-1/self.num_foot*(foot_forward_xy * base_forward_xy.view(self.num_envs, 1, 2)).abs().sum((1,2))))
        # .sub(foot_forward_xy).square().sum(dim=[1,2])
        # torch.square(foot_forward_xy.mean(dim=1,keepdim=True)-foot_forward_xy).sum(dim=[1,2])

        step_height = (self.foot_height + self.foot_height_offset)*foot_no_contact
        foot_height_rew = step_height.clamp_max(self.foot_height_clamp_max).sum(dim=-1) # reward foot height whenever
        # foot_height_rew = ((~self.contact_target)*step_height).clamp_max(self.foot_height_clamp_max).sum(dim=-1) # only reward at contact_target==0

        # #  foot_height_rew = (step_height.clamp_max(self.foot_height_clamp_max) - torch.clamp_min(step_height-self.foot_height_clamp_max,min=0)).clamp_min(0).sum(dim=-1)
        # # #shaped like: /\
        # foot_height_rew = (step_height.clamp_max(self.foot_height_clamp_max) - torch.clamp_min(step_height-self.foot_height_clamp_max,min=0)).clamp_min(0).sum(dim=-1)

        # self.rew_foot_height_exp_scale = 10
        # foot_height_rew = exp_square_sum(step_height - self.foot_height_clamp_max,exp_scale=self.rew_foot_height_exp_scale)
        # foot_height_rew = abs_sum((step_height-self.foot_height_clamp_max).clamp(-self.foot_height_clamp_max,self.foot_height_clamp_max))
        # foot_height_rew = square_sum((step_height - self.foot_height_clamp_max) *self.foot_contact_filt)
        # foot_height_rew[self.is_zero_command] = - foot_height_rew[self.is_zero_command]
        if self.keep_still_at_zero_command:
            foot_height_rew[self.is_zero_command] = 0
            rew["air_time"][self.is_zero_command] = 0
            rew["stance_time"][self.is_zero_command] = 0

        rew["foot_height"] = foot_height_rew

        if self.guided_contact:

            # should_swing = ~self.contact_target
            # phase_swing = self.phase[should_swing]
            # step_height_target = torch.zeros(self.num_envs,self.num_foot, dtype=torch.float, device=self.device)
            # step_height_target[should_swing] = self.foot_height_coeff[0]*(phase_swing**2) + self.foot_height_coeff[1]*(phase_swing**3) + self.foot_height_coeff[2]*(phase_swing**4)
            # if self.keep_still_at_zero_command:
            #     step_height_target[self.is_zero_command] = 0
            # self.rew_foot_height_exp_scale = -100
            # rew["foot_height"]  = exp_square_sum(step_height-step_height_target,exp_scale=self.rew_foot_height_exp_scale)

            rew["should_contact"] = torch.eq(self.foot_contact_filt,self.contact_target).mean(dim=1,dtype=torch.float)
            # rew["should_contact"][zero_command] = 1 # set contstant for zero command
            # rew["should_contact"][self.is_zero_command] = 0 # bad! do not use
            # rew["should_contact"][self.is_zero_command] =  0.8 + 0.2*rew["should_contact"][self.is_zero_command]

            # joints that are supposed in the swing/stance phase bool [num_envs, num_joints]
            # joint_stance = torch.any((self.contact_target).unsqueeze(2) & self.leg_to_dof_mask, dim=1)[:,self.actuated_dof_mask]
            joint_stance = torch.any((self.foot_contact_filt).unsqueeze(2) & self.leg_to_dof_mask, dim=1)[:,self.actuated_dof_mask] # use actual contact instead of target
            joint_swing = ~joint_stance

            rew["dof_force_target_swing"] = torch.exp(
                (joint_swing*self.actuated_dof_force_target.abs()*self.rew_dof_force_target_exp_scale).sum(dim=-1))

            if self.enable_passive_dynamics:
                rew["passive_action"] = torch.sum((1-self.action_is_on), dim=-1,dtype=torch.float)

                self.action_is_on_rate = (self.action_is_on - self.last_action_is_on) * self.rl_dt_inv

                rew["passive_action_rate"] = square_sum(self.action_is_on_rate,dim=-1)

        # log episode reward sums
        for key in rew.keys():
            self.episode_sums[key] += rew[key]  # unscaled
            rew[key] *= self.rew_scales[key]

        # total reward
        stacked_rewards = torch.stack(list(rew.values()), dim=0)
        self.rew_buf = torch.sum(stacked_rewards, dim=0)
        self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None) # NOTE THAT WE SCALE IT BY rl_dt
        # self.rew_buf = torch.clip(self.rew_buf*self.rl_dt, min=0.0, max=None) # NOTE THAT WE SCALE IT BY rl_dt
        
        # rew_group_sums =[torch.stack([rew[key] for key in group],dim=0).sum(dim=0) for group in self.rew_group]
        # rew_buf = rew_group_sums[0].clone()
        # for sum in rew_group_sums[1:]:
        #     rew_buf *= sum
        # self.rew_buf = torch.clip(rew_buf* self.rl_dt, min=0.0, max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        if self.enable_udp:  # send UDP info to plotjuggler
            dof_pow = (self.dof_vel[:,self.actuated_dof_mask] * self.actuated_dof_force_target)
            dof_pow_sum = dof_pow.abs().sum(dim=1)
            self.buffer_com_pos.add_and_fill_batch(self.root_state[:, :3])
            self.buffer_dof_pow.add_and_fill_batch(dof_pow_sum.unsqueeze(1))

            total_energy = torch.sum(self.buffer_dof_pow.storage,dim=0).view(-1) * self.rl_dt
            distance_traveled = (self.buffer_com_pos[0] - self.buffer_com_pos[-1])[:,:2].norm(dim=-1)

            cost_of_transport = total_energy/(torch.clamp_min(distance_traveled, 0.01)*self.total_gravity)

            data = {
                "t": self.control_steps*self.rl_dt,
                # "step_height_target": step_height_target,
                "cot": cost_of_transport,
                "action": self.action,
                "action_is_on": self.action_is_on,
                "action_is_on_rate": self.action_is_on_rate,
                "action_rate": self.action_rate,
                # "dof_jerk": self.dof_jerk[:,self.actuated_dof_mask],
                "dof_acc": self.dof_acc[:,self.actuated_dof_mask],
                "dof_vel": self.dof_vel[:,self.actuated_dof_mask],
                "dof_pos": self.dof_pos[:,self.actuated_dof_mask],
                "dof_pos_target": self.dof_pos_target,
                "dof_force_target": self.actuated_dof_force_target,
                "dof_force": self.dof_force[:,self.actuated_dof_mask],
                "dof_pow": dof_pow,
                "base_lin_vel": self.base_lin_vel,
                "base_ang_vel": self.base_ang_vel,
                "base_height": self.base_height,
                "foot_height": step_height,
                "projected_gravity": self.projected_gravity,
                "time_air": self.air_time,
                "time_stance": self.stance_time,
                "foot_pos": self.foot_pos_rel_yaw,
                "contact": self.foot_contact_filt,
                "phase": self.phase,
                "contact_target":self.contact_target,
                "rew_buf": self.rew_buf * self.rl_dt_inv,
                "commands": self.commands,
                "rew": {key: rew[key] * self.rl_dt_inv for key in rew},
                "rew_rel":{key: rew[key]/self.rew_buf for key in rew},
                "foot_rb_state": self.rb_state[:,self.foot_ids],
                # "foot_quat": self.foot_quat,
                "base_quat": self.base_quat,
                "root_state": self.root_state,
                "base_forward": self.base_forward,
                "foot_forward": self.foot_forward,
                "projected_gravity_filtered": self.projected_gravity_filtered,
                # "obs_buf":self.obs_buf,
                # "cam_pos": self.cam_pos,
            }
              
            if self.items_to_publish is not None:
                data = {key: data[key] for key in self.items_to_publish}
            self.data_publisher.publish({self.data_root_label:data})  
            
            # self.sim2real_publisher.publish({
            #     "base_ang_vel": self.base_ang_vel[0],
            #     "projected_gravity": self.projected_gravity[0],
            #     "dof_pos": self.dof_pos[0,self.actuated_dof_mask],
            #     "dof_vel": self.dof_vel[0,self.actuated_dof_mask],
            #     "action": self.action[0],
            #     "contact": self.foot_contact_filt[0],
            #     "contact_target":self.contact_target[0],
            #     "dof_pos_target": self.dof_pos_target[0],
            #     "phase": self.phase[0],
            #     "phase_sin_cos": self.phase_sin_cos[0],
            #     "cmd": self.commands[0][:3],
            #     "obs_buf":self.obs_buf[0],
            # })
    

    def reset_idx(self, env_ids: torch.Tensor):
        """Resets the specified environments."""
        len_ids = env_ids.numel()
        if len_ids == 0:
            return
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env_ids_raw = gymtorch.unwrap_tensor(env_ids_int32)
        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_state[env_ids] = self.base_init_state
            self.root_state[env_ids, :3] += self.env_origins[env_ids]
            self.root_state[env_ids, :2] += torch.empty(len_ids, 2, device=self.device, dtype=torch.float).uniform_(*self.randomize_base_init_pos_xy_range)
        else:
            self.root_state[env_ids] = self.base_init_state
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_state_raw, env_ids_raw, len_ids)

        if self.randomize_init_dof_pos:
            # dof_pos_offset = torch_rand_float(*self.randomize_init_dof_pos_range, (len_ids, self.num_dof), self.device)
            dof_pos_offset = torch.empty(len_ids, self.num_dof,dtype=torch.float, device=self.device).uniform_(*self.randomize_init_dof_pos_range)
            self.dof_pos[env_ids] = self.init_dof_pos[env_ids] + dof_pos_offset
        else:
            self.dof_pos[env_ids] = self.init_dof_pos[env_ids]
        
        if self.randomize_init_dof_vel:
            # self.dof_vel[env_ids] = torch_rand_float(*self.randomize_init_dof_vel_range, (len_ids, self.num_dof), self.device)
            self.dof_vel[env_ids] = torch.empty(len_ids, self.num_dof,dtype=torch.float, device=self.device).uniform_(*self.randomize_init_dof_vel_range)
        else:
            self.dof_vel[env_ids,:] = 0
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_raw, env_ids_raw, len_ids)

        temp_vec = torch.empty(len_ids, device=self.device, dtype=torch.float)
        # vx
        self.commands[env_ids, 0]= temp_vec.uniform_(*self.command_x_range)
        # vy
        self.commands[env_ids, 1]= temp_vec.uniform_(*self.command_y_range)
        # heading
        self.commands[env_ids, 3]= temp_vec.uniform_(*self.command_yaw_range)
        # # set small commands to zero # TODO CHANGE BACK
        # self.commands[env_ids] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > self.command_zero_threshold
        # ).unsqueeze(1)

        # SET COMMANDS TO ZERO FOR A PAERCENTAGE OF ENVIRONMENTS 
        if self.command_zero_probability:
            self.commands[env_ids[temp_vec.uniform_()<self.command_zero_probability], :] = 0 # the first 10% of the envs command will be zero

        # set small commands to zero
        self.is_zero_command[env_ids] = square_sum(self.commands[env_ids, :3], dim=1) < self.command_zero_threshold
        self.commands[self.is_zero_command]=0

        if self.enable_erfi: 
            self.erfi_rao[env_ids] = torch.empty(len_ids, self.num_actuated_dof,dtype=torch.float, device=self.device).uniform_(*self.erfi_rao_range)

        if self.randomize_dof_strength:
            self.dof_strength[env_ids] = torch.empty(len_ids, self.num_actuated_dof,dtype=torch.float, device=self.device).uniform_(*self.dof_strength_range)

        if self.enable_passive_dynamics:
            self.duration_since_action_switch[env_ids]=0
            self.last_action_is_on[env_ids] = 1 # fully active at first

        # # TODO: reset some observations
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.base_gravity_vec[env_ids])
        self.projected_gravity_filtered[env_ids]=self.projected_gravity[env_ids]

        if self.randomize_body_force:
            # reset rigid body forces
            self.rb_forces[env_ids, :, :] = 0.0
            self.random_force_prob[env_ids] = torch.empty(len_ids,dtype=torch.float, device=self.device).uniform_(*self.force_log_prob_range).exp_()

        self.last_foot_contact[env_ids] = 0
        self.foot_multi_contact_time[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.action_filt[env_ids] = 0.0
        # self.dof_pos_filt[env_ids] = self.dof_pos[env_ids]
        self.last_foot_contact_force[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_dof_acc[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.stance_time[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
    
        # reset delayed observations
        self.batched_obs_buf.reset_batch(env_ids)
        if self.asymmetric_obs:
            self.batched_states_buf.reset_batch(env_ids)
        
        # #reset action buffer
        # self.batched_action_buf.reset_batch(env_ids)
        
        # if self.enable_udp: 
        #     # strictly this should be reset, it's ok to ommit it for steady state values
        #     self.buffer_com_pos.reset_batch(env_ids)
        #     self.buffer_dof_pow.reset_batch(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            raw_sum = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s  # rewards per second
            self.extras["episode"][f'rew_{key}_raw'] = raw_sum * self.rl_dt  # scaled by policy dt
            self.extras["episode"][f'rew_{key}'] = raw_sum * self.rew_scales[key]
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = self.terrain_level_mean
        self.extras["episode"]["heights_curriculum_ratio"] = self.heights_curriculum_ratio

    def update_terrain_level(self, env_ids):
        """Updates the terrain level for curriculum learning."""
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_state[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        command_distance = torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s
        # # not timed out
        # self.terrain_levels[env_ids] -= 1 * ((distance < command_distance * 0.25) & (~self.timeout_buf[env_ids]))
        self.terrain_levels[env_ids] -= 1 * ((distance < command_distance * 0.5) & (~self.timeout_buf[env_ids]))
                                             
        # TODO check level up/down condition
        # self.terrain_levels[env_ids] += 1 * (distance > command_distance*0.5)
        # self.terrain_levels[env_ids] += 1 * torch.logical_or(
        #     distance > self.terrain.env_length / 2, distance > command_distance * 0.8)
        self.terrain_levels[env_ids] += 1 * torch.logical_or(
            distance > self.terrain.env_length / 2, distance > command_distance * 0.9)

        # # if reached max level, go back to level 0
        # self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows

        # if reached max level, go to a random level
        self.terrain_levels[env_ids] = self.terrain_levels[env_ids].clip(0)
        reached_max_levels = env_ids[self.terrain_levels[env_ids]>=self.terrain.env_rows]
        self.terrain_levels[reached_max_levels]=torch.randint_like(self.terrain_levels[reached_max_levels],low=0,high=self.terrain.env_rows)

        self.env_origins[env_ids] = self.terrain.env_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.terrain_level_mean = self.terrain_levels.float().mean()

    def push_robot_base(self):
        """Applies random pushes to the robots."""
        self.root_state[:, 7:13]+= torch_rand_tensor(
            self.push_vel_min, self.push_vel_max, (self.num_envs, 6), device=self.device
        )  # lin vel x/y/z
        self.gym.set_actor_root_state_tensor(self.sim, self.root_state_raw)

    def push_robot_base_indexed(self, env_ids: torch.Tensor):
        """Applies random pushes to the robots."""
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.root_state[env_ids, 7:13]+=torch_rand_tensor(
            self.push_vel_min, self.push_vel_max, (len(env_ids), 6), device=self.device
        )  # lin vel x/y/z
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.root_state_raw, gymtorch.unwrap_tensor(env_ids.type(torch.int32)), env_ids.numel())
    def push_rigid_bodies(self):
        """Applies random pushes to the rigid bodies."""
        pass

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            events = self.gym.query_viewer_action_events(self.viewer)
            for evt in events:
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames
                elif evt.action == "toggle_viewer_follow" and evt.value > 0:
                    self.viewer_follow = not self.viewer_follow
                elif evt.action == "reset" and evt.value>0:
                    # self.data_publisher.enable= not self.data_publisher.enable
                    # reset
                    self.progress_buf[:]= self.max_episode_length
                elif evt.action == "ref_env-" and evt.value > 0:
                    self.ref_env = (self.ref_env-1)%self.num_envs
                elif evt.action == "ref_env+" and evt.value > 0:
                    self.ref_env = (self.ref_env+1)%self.num_envs
            if self.enable_keyboard_operator:
                for evt in events:
                    if evt.action == "vx+" and evt.value > 0:
                        self.keyboard_operator_cmd[0] += 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vx-" and evt.value > 0:
                        self.keyboard_operator_cmd[0] -= 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vy+" and evt.value > 0:
                        self.keyboard_operator_cmd[1] += 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vy-" and evt.value > 0:
                        self.keyboard_operator_cmd[1] -= 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "heading+" and evt.value > 0:
                        self.keyboard_operator_cmd[2] += 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "heading-" and evt.value > 0:
                        self.keyboard_operator_cmd[2] -= 0.05
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "v=0" and evt.value > 0:
                        self.keyboard_operator_cmd[:] = 0
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "push" and evt.value > 0: # TODO fix CPU crashing in push_robots here
                        self.push_robot_base_indexed(torch.arange(self.num_envs, device=self.device))
                    # elif evt.action == "m+" and evt.value > 0:
                        # for i in range(self.num_envs):
                        #     env_handle = self.envs[i]
                        #     actor_handle = self.actor_handles[i]
                            # body_props = self.actor_rigid_body_properties[i]
                            # body_props[self.base_id].mass += 0.05
                            # self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=False)
                        # force = torch.zeros(self.num_envs, 3, device=self.device)
                        # self.gym.apply_rigid_body_force_tensors(self.sim,, gymapi.ENV_SPACE)
                        # print("mass increased by 0.05 kg")
                
                if self.data_receiver_data_id != self.data_receiver.data_id:
                    self.data_receiver_data_id = self.data_receiver.data_id
                    if "cmd" in self.data_receiver.data:
                        self.keyboard_operator_cmd[:] = torch.tensor(self.data_receiver.data["cmd"],device=self.device)
                    print(f"keybaord cmd:{self.keyboard_operator_cmd}")
                    if "reset" in self.data_receiver.data and self.data_receiver.data["reset"] == True:
                        self.progress_buf[:]= self.max_episode_length
                    if "push" in self.data_receiver.data and self.data_receiver.data["push"] == True:
                        self.push_robot_base_indexed(torch.arange(self.num_envs, device=self.device))
                        # self.data_publisher.enable = not self.data_publisher.enable
                # self.commands[:, [0, 1, 3]] = self.keyboard_operator_cmd
                self.commands[:, :3] = self.keyboard_operator_cmd

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.draw_debug_lines()
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(
                    self.viewer, os.path.join(self.record_frames_dir, f"frame_{self.control_steps}.png"))

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)
        
            # do modify camera position if viewer_follow
            if self.viewer_follow:
                self.cam_target_pos = self.root_state[self.ref_env, :3].clone()
                self.cam_pos = self.viewer_follow_offset+self.cam_target_pos
                # self.gym.viewer_camera_look_at(
                #     self.viewer, self.envs[self.ref_env], gymapi.Vec3(*self.cam_pos.cpu()), gymapi.Vec3(*self.cam_target_pos.cpu())) 
                cam_target_pos_filtered = self.cam_target_pos_filter_buffer.add(self.cam_target_pos)
                cam_pos_filtered = self.cam_pos_filter_buffer.add(self.cam_pos)
                self.gym.viewer_camera_look_at(
                    self.viewer, self.envs[self.ref_env], gymapi.Vec3(*cam_pos_filtered.cpu()), gymapi.Vec3(*cam_target_pos_filtered.cpu())) 

        return

    def draw_debug_lines(self):
        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        if self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            # self.gym.refresh_rigid_body_state_tensor(self.sim)

            # visualizing command 
            viz_cmd_start_point = torch.clone(self.root_state[:, :3])  # base pos
            # viz_cmd_start_point[:,2]+=0.5

            viz_cmd_xy_endpoint = torch.zeros(size=(self.num_envs, 3), dtype=torch.float, device=self.device)
            viz_cmd_xy_endpoint[:, :2] = self.commands[:, :2]
            viz_cmd_xy_endpoint = (viz_cmd_start_point+quat_apply_yaw(
                self.base_quat, viz_cmd_xy_endpoint)) # scaled
            
            viz_cmd_yaw_endpoint = torch.clone(viz_cmd_start_point)
            viz_cmd_yaw_endpoint[:, 2] += self.commands[:, 2]
            
            verts = torch.column_stack([viz_cmd_start_point, viz_cmd_xy_endpoint, 
                                        viz_cmd_start_point, viz_cmd_yaw_endpoint]).view((-1, 12)).cpu().numpy().view(dtype=gymapi.Vec3.dtype)
            colors_vel_dir = np.array([(1, 1, 0), (1, 1, 0)],dtype=gymapi.Vec3.dtype)

            for i, env in enumerate(self.envs):
                self.gym.add_lines(self.viewer, env, colors_vel_dir.shape[0], verts[i], colors_vel_dir)
                sphere_pose = gymapi.Transform(verts[i,1], r=None)
                gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, env, sphere_pose)
                sphere_pose = gymapi.Transform(verts[i,3], r=None)
                gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, env, sphere_pose)

            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_state[:, :3]
            ).unsqueeze(1)
            self._all_height_points_xy[:, :self.num_height_points,:] = points[:, :, :2]
            self._all_height_points_xy[:, self.num_height_points:,:] = self.rb_state[:,:, :2]

            xy = self._all_height_points_xy.cpu().numpy()
            z = self.heights_absolute.cpu().numpy()
            for i in range(self.num_envs):  # draw height points
                 for j in range(xy.shape[1]):
                    sphere_pose = gymapi.Transform(gymapi.Vec3(xy[i,j,0], xy[i,j,1], z[i,j]), r=None)
                    if j<=self.num_height_points:
                        gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                    else:
                        gymutil.draw_lines(self.sphere_geom_alt_color, self.gym, self.viewer, self.envs[i], sphere_pose)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # # randomize actions # TODO: rurrently need to completely bypass this
        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        self.action = torch.clamp(actions.clone().to(self.device), -self.clip_actions, self.clip_actions)
        # self.batched_action_buf.add_and_fill_batch(self.actions)

        # self.action_filt[:] = self.action_filt * 0.1 + self.action * 0.9

        # self.action_to_use = self.action_filt
        
        if self.randomize_action_delay:
            self.action_delay.uniform_(*self.action_delay_log_range).exp_()
            self.action_to_use = self.action * (1 - self.action_delay) + self.action_delay * self.action_to_use
        else:
            self.action_to_use = self.action

        if self.enable_passive_dynamics:
            self.dof_pos_target = self.action_scale * self.action_to_use[:,:self.num_actuated_dof] + self.default_dof_pos[:,self.actuated_dof_mask]
            self.action_is_on = sigmoid_k(self.action_to_use[:,self.num_actuated_dof:], self.action_is_on_sigmoid_k)
            if self.passive_curriculum:  # TODO change the hardcoded values to a variable
                # self.passive_action_min_value = 1.0 - float(min(self.common_step_counter/5e4,1.0))
                # self.passive_action_min_value = 0.8 - float(min(self.common_step_counter/5e4,0.8))
                # self.passive_action_min_value = 0.5 - float(min(self.common_step_counter/5e4,0.5))
                
                self.passive_action_min_value = max(0.5-self.common_step_counter/5e4,self.min_action_is_on)

                # self.passive_action_min_value = 0.5 - float(min(self.common_step_counter/5e4,0.5)) # always 10% on
                # self.action_is_on = torch.clamp_min(self.action_is_on,min=self.passive_action_min_value)
                self.action_is_on = self.passive_action_min_value+(1-self.passive_action_min_value)*self.action_is_on # alternative

                if self.common_step_counter % 1000 == 0:
                    print(f"self.passive_action_min_value={self.passive_action_min_value}")
            else:
                self.action_is_on = self.min_action_is_on + (1 - self.min_action_is_on) * self.action_is_on
            # self.action_is_on = self.actions_to_use[:,self.num_actuated_dof:]  > -0.5
            # self.actions[:, self.num_actuated_dof:] = torch.where(self.action_is_on, 1, -1)
        else:
            self.dof_pos_target = self.action_scale * self.action_to_use + self.default_dof_pos[:,self.actuated_dof_mask]
            self.action_is_on = 1

        # # soft limit clamping
        # below_min_limit = (self.dof_pos < self.dof_soft_limit_lower) & (self.dof_pos_target < self.dof_soft_limit_lower)
        # above_max_limit = (self.dof_pos > self.dof_soft_limit_upper) & (self.dof_pos_target > self.dof_soft_limit_upper)
        # torch.where(condition=below_min_limit,input=self.dof_soft_limit_lower,other=self.dof_pos_target,out=self.dof_pos_target)
        # torch.where(condition=above_max_limit,input=self.dof_soft_limit_upper,other=self.dof_pos_target,out=self.dof_pos_target)

        # apply actions
        self.pre_physics_step()
        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        
        # #TODO currently need to completely bypass this
        # # step physics and render each frame
        # for i in range(self.control_freq_inv):
        #     if self.force_render:
        #         self.render()
        #     self.gym.simulate(self.sim)
        if self.force_render:
            self.render()
        
        self.control_steps += 1

        # # randomize observations #TODO currently need to completely bypass dr_randomizations
        # if self.dr_randomizations.get('observations', None):
        #     self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras


    def pre_physics_step(self):
        """PD position control"""

        if self.randomize_body_force:
            self.rb_forces *= self.force_decay
            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero().ravel()
            if force_indices.numel() > 0:
                self.rb_forces[force_indices, :, :] = self.rb_forces[force_indices, :, :].uniform_(-1.0,1.0)* self.rb_force_mags[force_indices]
            # random force perturbation
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

        if self.enable_erfi: # extended random force injection
            self.erfi_rfi.uniform_(*self.erfi_rfi_range).add_(self.erfi_rao)
        for i in range(self.decimation):
            torque_computed = self.kp * (self.dof_pos_target - self.dof_pos) - self.kd * self.dof_vel
            torque_computed.clamp_(self.actuated_dof_force_min, self.actuated_dof_force_max)
            if self.enable_passive_dynamics:
                torque_computed*=self.action_is_on
            toque_actual = torque_computed*self.dof_strength+self.erfi_rfi  # randomized dof strength # TODO: make it better
            if i==0: # use the fist iteration FOR SIM2RAL maching
                self.actuated_dof_force_target[:] = torque_computed # this is the idealized computed torque without motor strength/randomization
                self.actuated_dof_force_target_actual[:] = toque_actual
            self.dof_actuation_force[:] = toque_actual # need to [:] to use the in-place version
            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(toque_actual))
            self.gym.set_dof_actuation_force_tensor(self.sim, self.dof_actuation_force_tensor)
            self.gym.simulate(self.sim)
            if self.device == 'cpu': # must fetch after simulate for any tensor.
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_rigid_body_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def pre_physics_step_with_passive_dof(self):
    # def pre_physics_step(self):
        """PD position control"""
        if self.enable_erfi: # extended random force injection
            self.erfi_rfi.uniform_(*self.erfi_rfi_range).add_(self.erfi_rao)
        for i in range(self.decimation):
            torque = self.kp * (self.dof_pos_target - self.dof_pos[:,self.actuated_dof_mask]) - self.kd * self.dof_vel[:,self.actuated_dof_mask]
            if self.enable_passive_dynamics:
                torque*=self.action_is_on
            torque.add_(self.erfi_rfi).clamp_(self.actuated_dof_force_min, self.actuated_dof_force_max)
            # TODO maybe check if action exceeds limit and make it a reward
            self.dof_actuation_force[:,self.actuated_dof_mask] = torque
            self.gym.set_dof_actuation_force_tensor(self.sim, self.dof_actuation_force_tensor)
        
            if self.num_marker_pairs > 0:
                marker_pos_error = self.rb_state[:,self.marker_pair_ids[:,1],:3] -  self.rb_state[:,self.marker_pair_ids[:,0],:3] # [num_envs, num_marker_pairs, 3]
                marker_pos_error_norm = torch.linalg.vector_norm(marker_pos_error, ord=2, dim=-1, keepdim=True) # [num_envs, num_marker_pairs, 1]
                
                marker_force = 100000 * (marker_pos_error_norm-self.marker_pair_l0.view(1,self.num_marker_pairs,1))*marker_pos_error/marker_pos_error_norm
                forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                forces[:,self.marker_pair_ids[:,0]] = marker_force
                forces[:,self.marker_pair_ids[:,1]] = -marker_force
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
      
            self.gym.simulate(self.sim)
            if self.device == 'cpu': # must fetch after simulate for any tensor.
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.actuated_dof_force_target[:] = torque

        # self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        # self.gym.refresh_rigid_body_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)


    def post_physics_step(self):

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        # if self.common_step_counter % self.push_interval == 0 and self.should_push_robots:
        #     self.push_robots()

        if self.should_push_robots:
            env_ids = torch.nonzero(self.progress_buf%self.push_interval == 0).squeeze_(1)
            if env_ids.numel()>0:
                self.push_robot_base_indexed(env_ids)
    
        # prepare quantities
        # self.base_quat = self.root_state[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.base_gravity_vec)

                
        if self.randomize_projected_gravity_delay:
            self.projected_gravity_delay.uniform_(*self.projected_gravity_delay_log_range).exp_()
            self.projected_gravity_filtered[:] = normalize(self.projected_gravity_delay*self.projected_gravity_filtered + (1 - self.projected_gravity_delay)*self.projected_gravity)
        else:
            self.projected_gravity_filtered[:] = self.projected_gravity

            
        self.base_forward = quat_apply(self.base_quat, self.base_forward_vec_local)
        heading = torch.atan2(self.base_forward[:, 1], self.base_forward[:, 0])

        self.base_quat_yaw = get_quat_yaw(self.base_quat)

        self.dof_acc = (self.dof_vel - self.last_dof_vel) * self.rl_dt_inv  # TODO check if [:] is needed # TODO
        # self.dof_jerk = (self.dof_acc - self.last_dof_acc) * self.rl_dt_inv

        self.foot_quat = self.rb_state[:, self.foot_ids, 3:7] # foot quaternion [num_envs, num_foot, 4] # this is a copy!!
        self.foot_projected_gravity=quat_rotate_inverse(self.foot_quat.view(-1, 4), self.foot_gravity_vec).view(self.num_envs, self.num_foot, 3)
        
        self.foot_forward = quat_apply(self.foot_quat.view(-1, 4), self.foot_forward_vec_local).view(self.num_envs, self.num_foot, 3)

        # # foot_quat relative to the base yaw frame: foot_quat * base_quat_yaw_inv
        # base_quat_yaw_inv = quat_conjugate(self.base_quat_yaw) # 1/quat_yaw
        # self.foot_quat_rel = quat_mul(self.foot_quat.view(-1, 4), base_quat_yaw_inv.repeat_interleave(self.num_foot, dim=0)).view(self.num_envs, self.num_foot, 4) # foot quat relative to the base yaw frame
        # self.foot_forward_rel = quat_apply(self.foot_quat_rel.view(-1, 4), self.foot_forward_vec_local).view(self.num_envs, self.num_foot, 3)

        self.foot_pos = self.rb_state[:, self.foot_ids, 0:3]
        # relative foot position in yaw frame
        foot_pos_rel =  self.foot_pos - self.root_state[:, :3].view(self.num_envs, 1, 3)
        
        self.foot_pos_rel_yaw =  quat_rotate_inverse(self.base_quat_yaw.repeat_interleave(self.num_foot, dim=0), 
                                   foot_pos_rel.view(-1, 3)).view(self.num_envs, self.num_foot, 3)
        
        self.foot_lin_vel = self.rb_state[:, self.foot_ids, 7:10]

        # foot contact
        # self.contact = torch.norm(contact_forces[:, foot_indices, :], dim=2) > 1.
        self.foot_contact_force = self.contact_force[:, self.foot_ids, :]
        self.foot_contact = self.foot_contact_force[:, :, 2] > self.foot_contact_threshold  # todo check with norm
        
        # # HACK: no contact
        # self.foot_contact[:] = 0

        self.foot_contact_filt = torch.logical_or(self.last_foot_contact, self.foot_contact) # filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.last_foot_contact = self.foot_contact
        # self.foot_contact = self.sensor_forces[:, :, 2] > 1.0

        if self.viewer and self.enable_keyboard_operator:
            self.commands[:, 3] = wrap_to_pi(2 * self.commands[:, 2] + heading)
        else:
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
        
        # set zero command if the magnitude is too small
        self.is_zero_command[:] = square_sum(self.commands[:, :3], dim=1) < self.command_zero_threshold
        self.commands[self.is_zero_command]=0

        if self.guided_contact: # update phase for the contact sequence
            self.update_phase()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_observations()
        self.compute_reward()

        # update last_...

        if self.enable_passive_dynamics:
            self.last_action_is_on = self.action_is_on

        # self.dof_pos_filt[:] = self.dof_pos_filt * 0.97 + self.dof_pos * 0.03
        self.last_action[:] = self.action
        self.last_dof_vel[:] = self.dof_vel
        self.last_dof_acc[:] = self.dof_acc
        self.last_foot_contact_force[:] = self.foot_contact_force

        # resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)        


    def init_height_points(self):
        """
        initialize height points in cpu, save self.num_height_points and self.height_points
        self.height_points[:,:self.num_height_points,0] is grid_x
        self.height_points[:,:self.num_height_points,1] is grid_y
        self.num_height_points[:,-1,:] is base (0,0,0)
        """
        cfg_heightmap = self.cfg["env"]["heightmap"]
        x = torch.tensor(cfg_heightmap["x"], dtype=torch.float, device=self.device)
        y = torch.tensor(cfg_heightmap["y"], dtype=torch.float, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device)
        # num_envs, num_points_per_env, xyz
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        self.height_points = points
        # num_envs, num_points_per_env+num_bodies, xyz
        self.heights_absolute = torch.zeros(self.num_envs, self.num_height_points+self.num_bodies, device=self.device, dtype=torch.float)
        self.heights_relative = torch.empty_like(self.heights_absolute)
        self._all_height_points_xy = torch.zeros(self.num_envs, self.num_height_points+self.num_bodies, 2, device=self.device, dtype=torch.float)

        
    def get_heights(self):
        """get heights at sampled locations"""
        # if self.terrain_type == 'plane': # heights of the plane at sampled locations are all zero
        #     self.heights_absolute = torch.zeros(self.num_envs, self.num_height_points+self.num_bodies, device=self.device)
        if self.terrain_type != 'plane':
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_state[:, :3]
            ).unsqueeze(1)
            self._all_height_points_xy[:, :self.num_height_points,:] = points[:, :, :2]
            self._all_height_points_xy[:, self.num_height_points:,:] = self.rb_state[:,:, :2]
            # self.root_states: (num_env,13)
            # points: (num_env,num_points_per_env+1 (root_pos),3 (xyz))
            # ## points = torch.cat((points, self.root_states[:, :3].unsqueeze(1)), dim=1)
            self.heights_absolute = self.terrain.get_heights(self._all_height_points_xy).view(self.num_envs, -1)
            # heights_absolute: (num_env,num_points_per_env+1 (body_com))
        
        self.heights_relative[:,:self.num_height_points] = self.root_state[:, 2].unsqueeze(1) - self.heights_absolute[:,:self.num_height_points]
        self.heights_relative[:,self.num_height_points:] = self.rb_state[:,:, 2] - self.heights_absolute[:,self.num_height_points:]
        self.base_height: torch.Tensor = self.heights_relative[:, self.num_height_points+self.base_id]
        self.foot_height: torch.Tensor = self.heights_relative[:, self.num_height_points+self.foot_ids]

def get_matching_str(source, destination, case_sensitive=False, comment=""):
    """Finds case-insensitive partial matches between source and destination lists."""
    def find_matches(src_item):
        if case_sensitive:
            matches = [item for item in destination if src_item in item]
        else:
            matches = [item for item in destination if src_item.lower() in item.lower()]
        if not matches:
            raise KeyError(f"cannot locate {src_item}. [{comment}]")
        elif len(matches) > 1:
            raise KeyError(f"find multiple instances for {src_item}. [{comment}]")
        return matches[0]  # Return just the first match
    if isinstance(source, str):  # one to many
        if case_sensitive:
            matches = [item for item in destination if source in item]
        else:
            matches = [item for item in destination if source.lower() in item.lower()]  
        if not matches:
            raise KeyError(f"cannot locate {source} [{comment}\navailables are {destination}")
        return matches
    else:  # one to one     
        return [find_matches(item) for item in source]

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def get_quat_yaw(quat) -> torch.Tensor:
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_yaw


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def torch_rand_tensor(lower: torch.Tensor, upper: torch.Tensor, shape: Tuple[int, int], device: str) -> torch.Tensor:
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def square_sum(input: torch.Tensor,dim: int=-1) -> torch.Tensor:
    return torch.square(input).sum(dim=dim)

@torch.jit.script
def square_sum_clamp_max(input: torch.Tensor,dim: int=-1, max: float=1.0) -> torch.Tensor:
    return torch.square(input).sum(dim=dim).clamp_max_(max)

@torch.jit.script
def exp_square_sum(input: torch.Tensor,exp_scale: float, dim: int=-1) -> torch.Tensor:
    return torch.exp(torch.square(input).sum(dim=dim)*exp_scale)

@torch.jit.script
def exp_square(input: torch.Tensor,exp_scale: float) -> torch.Tensor:
    return torch.exp(torch.square(input)*exp_scale)

@torch.jit.script
def exp_weighted_square_sum(x: torch.Tensor, exp_scale: torch.Tensor,dim: int=-1):
    return torch.exp(torch.sum(exp_scale*x.square(), dim=dim))

@torch.jit.script
def out_of_bound_norm(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return torch.norm(input - torch.clamp(input, lower, upper), dim=dim)

@torch.jit.script
def out_of_bound_square_sum(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return torch.square(input - torch.clamp(input, lower, upper)).sum(dim=dim)


@torch.jit.script
def out_of_bound_abs_sum(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return (input - torch.clamp(input, lower, upper)).abs().sum(dim=dim)


@torch.jit.script
def out_of_float_bound_squared_sum(input: torch.Tensor, lower: float, upper: float, dim: int=-1) -> torch.Tensor:
    return torch.square(input - torch.clamp(input, lower, upper)).sum(dim=dim)


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
def log_unifrom(tensor:torch.Tensor,log_low:float, log_high:float):
    return tensor.uniform_(log_low, log_high).exp_()

@torch.jit.script
def reverse_bell(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    return 1 - 1 / (1 + torch.pow(torch.abs(x / a - c), b))

@torch.jit.script
def sigmoid_k(x: torch.Tensor, k: float) -> torch.Tensor:
    return 1 / (1 + torch.exp(-k*x))

## this script is derived from anymal_terrain.py in IsaacGymEnvs
# https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/anymal_terrain.py
# the orignal script contains the copyright notice as below
#
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
import numpy as np
import os
import sys
import time
import datetime
from typing import Dict, Any, Tuple
from operator import itemgetter
from gym import spaces
from collections.abc import Iterable

from isaacgym.torch_utils import get_axis_params, torch_rand_float, quat_rotate_inverse, quat_apply, normalize
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
from envs.common.utils import bcolors as bc
from envs.common.urdf_utils import get_leaf_nodes
from envs.common.publisher import DataPublisher, DataReceiver
from envs.common.terrain import Terrain
from isaacgym import gymutil

from hydra.utils import to_absolute_path

class LeggedTerrain(VecTask):
    """
    issaac gym envs for task "A1Terrain" and "AnymalTerrain"
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

        # self.sim_params = self._VecTask__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        self.sim_params = self._parse_sim_params()

        self.gym = gymapi.acquire_gym()

        # Creates the physics simulation and terrain.
        self.up_axis_idx = {"x": 0, "y": 1, "z": 2}[self.cfg["sim"]["up_axis"]]  # index of up axis: x=0, y=1, z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.load_asset()

        self.num_environments = self.cfg["env"]["numEnvs"]  # self.num_envs
        self.num_agents = self.cfg["env"].get("numAgents", 1)  # used for multi-agent environments

        # control
        self.kp: float = self.cfg["env"]["control"]["stiffness"]
        self.kd: float = self.cfg["env"]["control"]["damping"]
        self.dof_force_target_limit: float = self.cfg["env"]["control"]["limit"]  # Torque limit [N.m]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        # TODO: MAYBE SET A BETTER SCALE FOR DOF FORCES
        self.dof_force_scale = self.cfg["env"]["learn"].get("dofForceScale",1/self.dof_force_target_limit)
        self.dof_force_target_scale = self.cfg["env"]["learn"].get("dofForceTargetScale",1/self.dof_force_target_limit)
        self.heightmap_scale = self.cfg["env"]["learn"]["heightMapScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        # treat commends below this threshold as zero [m/s]
        self.command_zero_threshold = self.cfg["env"]["commandZeroThreshold"]

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
        self.named_init_dof_pos = self.cfg["env"].get("initialJointPositions", self.named_default_dof_pos)

        def get_dof_pos(value):
            if isinstance(value, dict):
                dof = torch.tensor(
                    itemgetter(*self.dof_names)(value), dtype=torch.float, device=self.device
                ).repeat(self.num_envs, 1)
            else: # list
                dof = torch.tensor(value, dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
                value = {n: v for n, v in zip(self.dof_names, value)}
            return dof,value
        
        self.default_dof_pos,self.named_default_dof_pos = get_dof_pos(self.named_default_dof_pos)
        self.desired_dof_pos,self.named_desired_dof_pos = get_dof_pos(self.named_desired_dof_pos)
        self.init_dof_pos,self.named_init_dof_pos = get_dof_pos(self.named_init_dof_pos)

        # update pos to default dof pos
        self.asset_urdf.update_cfg(self.named_default_dof_pos)
        lower_bound_z = self.asset_urdf.collision_scene.bounding_box.bounds[0,2]
        
        # target base height [m]
        self.target_base_height = self.cfg["env"].get("baseHeightTarget",None)

        if self.target_base_height is None:
            base_height_offset= self.cfg["env"].get("baseHeightOffset",0.1)
            self.target_base_height = -lower_bound_z
            self.cfg["env"]["baseInitState"]["pos"][2] = float(self.target_base_height+base_height_offset)
            print(f"{bc.WARNING}[infer from URDF] target_base_height = {self.target_base_height:.4f} {bc.ENDC}")
            print(f"{bc.WARNING}[infer from URDF] self.cfg['env']['baseInitState']['pos'][2] = {self.cfg['env']['baseInitState']['pos'][2]:.4f} {bc.ENDC}")


        # needed for foot height reward
        z_pos_foot = np.mean([self.asset_urdf.get_transform(foot_name,collision_geometry=True)[2,3] for foot_name in self.foot_names]) # z position of foot
        self.foot_height_offset = lower_bound_z - z_pos_foot
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
            # self.sim2real_publisher = DataPublisher(target_url="udp://localhost:9876")

        # reward scales
        cfg_rew: Dict[str, Any] = self.cfg["env"]["learn"]["reward"]


        self.rew_scales = {key: rew_item["scale"] for key,rew_item in cfg_rew.items()}
        for key in self.rew_scales:
            self.rew_scales[key] = float(self.rew_scales[key]) * self.rl_dt
        # do not scale termination reward
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]

        self.torque_penalty_bound = self.cfg["env"]["learn"].get("torquePenaltyBound", 0.0)
        print(f"torque penalty bound = {self.torque_penalty_bound}")

        # self.rew_lin_vel_xy_exp_scale: float = cfg_rew["lin_vel_xy"]["exp_scale"]
        # self.rew_lin_vel_z_exp_scale: float = cfg_rew["lin_vel_z"]["exp_scale"]
        # self.rew_ang_vel_z_exp_scale: float = cfg_rew["ang_vel_z"]["exp_scale"]

        self.rew_orient_exp_scale: float = cfg_rew["orientation"]["exp_scale"]
        self.rew_foot_orient_exp_scale: float = cfg_rew["foot_orientation"]["exp_scale"]

        
        self.rew_lin_vel_exp_scale: torch.Tensor =cfg_rew["lin_vel"]["exp_scale"] * \
            torch.tensor(cfg_rew["lin_vel"]["normalize_by"],dtype=torch.float, device=self.device)
        self.rew_ang_vel_exp_scale: torch.Tensor =cfg_rew["ang_vel"]["exp_scale"] * \
            torch.tensor(cfg_rew["ang_vel"]["normalize_by"],dtype=torch.float, device=self.device)
        

        self.rew_dof_force_target_exp_scale: float = cfg_rew["dof_force_target"]["exp_scale"]
        # rew_dof_force_target_exp_scale_normalize_by = cfg_reward["dof_force_target"]["normalize_by"]
        self.dof_effort_limit = torch.tensor(self.dof_props["effort"],dtype=torch.float, device=self.device)
        self.rew_dof_force_target_exp_scale= (self.rew_dof_force_target_exp_scale/ (self.dof_effort_limit*self.num_dof))

        self.max_foot_contact_force = 100  # [N] # todo refactor

        # base height reward: reverse bell shaped curve
        # https://researchhubs.com/post/maths/fundamentals/bell-shaped-function.html
        a, b = self.cfg["env"]["learn"].get("baseHeightRewardParams", [0.04, 3])
        self.base_height_rew_a, self.base_height_rew_b = float(a), float(b)

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
            self.erfi_rao = torch.empty(self.num_envs,self.num_dof, dtype=torch.float, device=self.device).uniform_(*self.erfi_rao_range)
            self.erfi_rfi = torch.empty(self.num_envs,self.num_dof, dtype=torch.float, device=self.device).uniform_(*self.erfi_rfi_range)
            self.pre_physics_step = self.pre_physics_step_erfi # override pre_physics_step with extended random force injection

        # heightmap
        self.init_height_points()

        # passive dynamics
        self.enable_passive_dynamics = self.cfg["env"].get("enablePassiveDynamics", False)

        infer_action = self.cfg["env"]["numActions"] == "infer"
        if infer_action:
            self.cfg["env"]["numActions"] = len(self.dof_names)
            if self.enable_passive_dynamics:
                self.cfg["env"]["numActions"] = len(self.dof_names)*2
        
        # observation dimensions of specific items
        self.obs_dim_dict = {
            "linearVelocity": 3,
            "angularVelocity": 3,
            "projectedGravity": 3,
            "commands": 3,  # vel_x,vel_y, vel_yaw, (excluding heading)
            "dofPosition": self.num_dof,
            "dofVelocity": self.num_dof,
            "dof_force_target": self.num_dof,
            "dofForce": self.num_dof,
            "heightMap": self.num_height_points,  # excluding the base origin measuring point
            "base_height": 1,
            "actions": self.cfg["env"]["numActions"],
            "contact": self.num_foot,  # foot contact indicator
            "phase": self.num_foot*2, # phase of each foot (contact sequece)
            "contactTarget": self.num_foot,  # foot contact indicator
        }
        self.obs_names = tuple(self.cfg["env"]["observationNames"])
        num_obs = np.sum(itemgetter(*self.obs_names)(self.obs_dim_dict))
        if self.cfg["env"]["numObservations"] == "infer":
            # TODO refactor this number to be automatic
            self.cfg["env"]["numObservations"] = self.num_observations = num_obs
            # self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
            # self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
            print(f"inferring, numObservations={num_obs}")
        assert self.cfg["env"]["numObservations"] == num_obs

        self.asymmetric_obs = self.cfg["env"].get("asymmetric_observations", False)
        if self.asymmetric_obs:
            self.state_names = tuple(self.cfg["env"]["stateNames"])
            num_state = np.sum(itemgetter(*self.state_names)(self.obs_dim_dict))
            self.cfg["env"]["numStates"] = self.num_states = num_state
        
        self.num_observations = self.cfg["env"]["numObservations"]
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

        # contact_force: (num_envs, num_bodies, xyz axis)
        self.contact_force: torch.Tensor = gymtorch.wrap_tensor(self.net_contact_force_raw).view(self.num_envs, -1, 3)

        # rb_state: (num_envs,num_rigid_bodies,13)
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13])
        self.rb_state: torch.Tensor = gymtorch.wrap_tensor(self.rb_state_raw).view(self.num_envs, -1, 13)

        # dof force tensor
        self.dof_force: torch.Tensor = gymtorch.wrap_tensor(self.dof_force_tensor_raw).view(self.num_envs, self.num_dof)

        # reward episode sums (unscaled)
        def torch_zeros(): return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_sums = {key: torch_zeros() for key in self.rew_scales.keys()}

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.noise_vec = torch.zeros_like(self.obs_buf, dtype=torch.float, device=self.device)

        # # delayed observations
        # from envs.common.buffer import BatchedRingTensorBuffer
        # self.max_obs_delay = 1
        # self.delayed_obs_buf = BatchedRingTensorBuffer(buffer_len=self.max_obs_delay,batch_size=self.num_envs, shape=self.num_obs,dtype=torch.float, device=self.device)


        # gravity_vec=[0,0,-1]
        gravity_vec = torch.tensor(get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device)
        self.gravity_vec = gravity_vec.repeat((self.num_envs, 1))
        self.foot_gravity_vec = gravity_vec.repeat((self.num_foot*self.num_envs, 1))

        self.forward_vec = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.dof_force_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        # self.actions_filt = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        # self.dof_pos_filt = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        # foot air time and stance time
        self.air_time = torch.zeros(self.num_envs, self.num_foot, dtype=torch.float, device=self.device)
        self.stance_time = torch.zeros(self.num_envs, self.num_foot, dtype=torch.float, device=self.device)
        self.last_foot_contact_force = torch.zeros(self.num_envs, self.num_foot, 3,
                                                   dtype=torch.float, device=self.device)
        
        # single contact reward parameters
        self.max_single_contact: int = cfg_rew["single_contact"]["max_single_contact"]
        self.foot_multi_contact_grace_period: float = cfg_rew["single_contact"]["grace_period"]
        self.foot_multi_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.last_dof_vel = torch.zeros_like(self.dof_vel)


        cfg_guided_contact = self.cfg["env"]["learn"].get("guided_contact", {})
        self.guided_contact = cfg_guided_contact.get("enable", False)
        self.phase=None
        self.phase_sin_cos=None
        self.contact_target=None
        if self.guided_contact:
            # normalized phase [0,1]
            self.phase_offset = torch.tensor(cfg_guided_contact["phase_offset"],dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            self.phase_freq: float = cfg_guided_contact["phase_freq"] # [Hz]
            self.phase_stance_ratio: float = cfg_guided_contact["phase_stance_ratio"] # [1]
            self.phase_swing_ratio = 1 - self.phase_stance_ratio
            self.update_phase()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True
        
        # rendering
        self.set_viewer()

    def update_phase(self):
        """update normalized contact phase for each foot, 
           stance phase: 0<phase<stance_ratio
           swing phase:  stance_ratio<phase<1
        """
        self.phase = (self.phase_freq * self.rl_dt * self.progress_buf.unsqueeze(1) +self.phase_offset) % 1
        self.phase_sin_cos = torch.column_stack((torch.sin(2*torch.pi*self.phase), torch.cos(2*torch.pi*self.phase))) # HACK TODO ADD 2PI
        self.contact_target = self.phase < self.phase_stance_ratio # 1 = stance, 0 = swing
        self.contact_target[self.is_zero_command] = 1 # set contact_target to 1 if zero command

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

        self.cam_pos = self.cfg["env"]["viewer"]["pos"]
        self.cam_target_pos = self.cfg["env"]["viewer"]["lookat"]
        self.ref_env:int = int(self.cfg["env"]["viewer"]["refEnv"])%self.num_envs

        self.viewer_follow = self.cfg["env"]["viewer"]["follow"]
        if self.viewer_follow:
            self.viewer_follow_offset = torch.tensor(self.cfg["env"]["viewer"].get("follower_offset", [0.6, 1.2, 0.4]))
            self.cam_target_pos = self.root_state[self.ref_env, :3].clone().cpu() 
            self.cam_pos = self.viewer_follow_offset.cpu() + self.cam_target_pos
        
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
            "commands": 0,
            "dofPosition": cfg_learn["dofPositionNoise"] * noise_level * self.dof_pos_scale,
            "dofVelocity": cfg_learn["dofVelocityNoise"] * noise_level * self.dof_vel_scale,
            "dof_force_target": 0,
            "dofForce": 0, # TODO, MAYBE ADD NOISE FOR DOF FORCE
            "heightMap": cfg_learn["heightMapNoise"] * noise_level * self.heightmap_scale,
            "base_height": 0,
            "actions": 0,  # previous actions
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

        self.num_dof = self.gym.get_asset_dof_count(self.asset)
        self.dof_names = self.gym.get_asset_dof_names(self.asset)

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
        dof_margin = self.cfg["env"]["learn"].get("dofLimitMargins", 0)
        dof_margin = torch.tensor(dof_margin, dtype=torch.float, device=self.device)
        self.dof_lower = torch.tensor(self.dof_props['lower'], dtype=torch.float, device=self.device) + dof_margin
        self.dof_upper = torch.tensor(self.dof_props['upper'], dtype=torch.float, device=self.device) - dof_margin

        def get_matching_str(source, destination, comment=""):
            """Finds case-insensitive partial matches between source and destination lists."""
            def find_matches(src_item):
                matches = [item for item in destination if src_item.lower() in item.lower()]
                if not matches:
                    raise KeyError(f"cannot locate {src_item}. [{comment}]")
                elif len(matches) > 1:
                    raise KeyError(f"find multiple instances for {src_item}. [{comment}]")
                return matches[0]  # Return just the first match
            if isinstance(source, str):  # one to many
                matches = [item for item in destination if source.lower() in item.lower()]  
                if not matches:
                    raise KeyError(f"cannot locate {source} [{comment}\navailables are {destination}")
                return matches
            else:  # one to one     
                return [find_matches(item) for item in source]

        # body
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self.body_names = self.gym.get_asset_rigid_body_names(self.asset)

        asset_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset)
        asset_rigid_body_id_dict = {value: key for key, value in asset_rigid_body_dict.items()}

        # body: base
        base_name = cfg_asset.get("baseName", None)
        if base_name is None:  # infer base_name
            base_name = self.asset_urdf.base_link
        else:
            base_name = get_matching_str(source=base_name, destination=self.body_names, comment="base_name")[0]
        self.base_id = asset_rigid_body_dict[base_name]
        
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
            # HACK: exclude base link and feet, include all other links
            exclude_link_names = set(self.foot_names)
            exclude_link_names.add(base_name)
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

        print(f"base = {base_name}: {self.base_id}")
        # print(f"hip = {dict(zip(hip_names,self.hip_ids.tolist()))}")
        print(f"knee = {dict(zip(knee_names,self.knee_ids.tolist()))}")
        print(f"foot = {dict(zip(self.foot_names,self.foot_ids.tolist()))}")
        # print(f"dof_hip = {dict(zip(dof_hip_names,self.dof_hip_ids.tolist()))}")
        assert self.base_id != -1
        # assert len(self.dof_hip_ids)==4
        # assert self.dof_hip_ids.tolist() == [0, 3, 6, 9]
        ####################
        
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
            self.base_pos_xy_range = (-1.0, 1.0)  # TODO refactor into config
            base_pos_xy_offset = torch.empty(self.num_envs, 2, dtype=torch.float,device=self.device).uniform_(*self.base_pos_xy_range)

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
                # pos[:2] += torch_rand_float(*self.base_pos_xy_range, (2, 1), self.device).squeeze(1)
                pos[:2] += base_pos_xy_offset[i]
                start_pose.p = gymapi.Vec3(*pos)
            actor_handle = self.gym.create_actor(
                env_handle, self.asset, start_pose, "actor", i, self.collision_filter, 0
            )
            # dof_force_tensor
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            if randomize_friction:
                for s in range(len(rigid_shape_prop)):
                    rigid_shape_prop[s].friction = friction_buckets[i]
                self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_prop)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, self.dof_props)
            
            if randomize_base_mass:
                body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
                body_props[self.base_id].mass += baseMass_buckets[i]
                self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def check_termination(self):
        """Checks if the episode should terminate."""
        # base tilt 45 degree or base contact with ground
        self.reset_buf = torch.logical_or(self.projected_gravity[:,2]>-0.7, square_sum(self.contact_force[:, self.base_id, :]) > 1.0)
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
        heights = torch.clip(self.heights_relative - self.target_base_height, -1.0, 1.0) * self.heightmap_scale

        obs_dict = {
            "linearVelocity": self.base_lin_vel * self.lin_vel_scale,
            "angularVelocity": self.base_ang_vel * self.ang_vel_scale,
            "projectedGravity": self.projected_gravity,
            "commands": self.commands[:, :3] * self.commands_scale,
            "dofPosition": self.dof_pos * self.dof_pos_scale,
            "dofVelocity": self.dof_vel * self.dof_vel_scale,
            "dof_force_target": self.dof_force_target * self.dof_force_target_scale,
            "dofForce": self.dof_force * self.dof_force_scale,
            "base_height": heights[:, self.num_height_points+self.base_id],
            "heightMap": heights[:, :self.num_height_points],
            "actions": self.actions,
            "contact": self.foot_contact,
            "contactTarget": self.contact_target,
            "phase": self.phase_sin_cos,
        }
        # update observation buffer
        self.obs_buf = torch.cat(itemgetter(*self.obs_names)(obs_dict), dim=-1)

        if self.asymmetric_obs: # update state buffer
            self.states_buf = torch.cat(itemgetter(*self.state_names)(obs_dict), dim=-1)

        if self.add_noise:
            self.noise_vec.uniform_(-1.0, 1.0).mul_(self.noise_scale_vec)  # scaled noise vector
            self.obs_buf += self.noise_vec

        # self.delayed_obs_buf.add_and_fill_batch(self.obs_buf)
        # self.obs_buf = self.delayed_obs_buf[-1]

    def compute_reward(self):
        """Computes the reward for the current state and action."""
        rew = {}
        # velocity tracking reward

        desired_vel = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float)
        desired_vel[:, [0,1,5]] = self.commands[:, [0,1,2]]
        # linear velocity error xyz
        lin_vel_error: torch.Tensor = desired_vel[:, :3] - self.base_lin_vel
        ang_vel_error: torch.Tensor = desired_vel[:, 3:] - self.base_ang_vel

        rew["lin_vel"] = torch.exp(torch.sum(self.rew_lin_vel_exp_scale*lin_vel_error.square(), dim=-1))
        rew["ang_vel"] = torch.exp(torch.sum(self.rew_ang_vel_exp_scale*ang_vel_error.square(), dim=-1))

        # lin_vel_error = square_sum(self.commands[:, :2] - self.base_lin_vel[:, :2])
        # ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # rew["lin_vel_xy"] = torch.exp(self.rew_lin_vel_xy_exp_scale*lin_vel_error)
        # rew["ang_vel_z"] = torch.exp(self.rew_ang_vel_z_exp_scale*ang_vel_error)

        # # other base velocity penalties
        # rew["lin_vel_z"] = torch.square(self.base_lin_vel[:, 2])
        # rew["ang_vel_xy"] = square_sum(self.base_ang_vel[:, :2])

        # orientation penalty
        # rew["orientation"] = square_sum(self.projected_gravity[:, :2])
        projected_gravity_xy = square_sum(self.projected_gravity[:, :2])
        rew["orientation"]=torch.exp(self.rew_orient_exp_scale*projected_gravity_xy)

        # base height penalty
        # rew["base_height"] = torch.square(self.heights_relative[:, self.num_height_points] - self.target_base_height)
        # exponential reward
        # base_height_error = torch.square(self.heights_relative[:, self.num_height_points] - self.target_base_height)
        # rew["base_height"] = 1.0 - torch.exp(-base_height_error / 0.001)
        # bell-shaped reward

        rew["base_height"] = reverse_bell(
            self.base_height, a=self.base_height_rew_a, b=self.base_height_rew_b, c=self.target_base_height
        )

        # torque penalty
        rew["dof_force_target"] = torch.exp((self.dof_force_target.abs()*self.rew_dof_force_target_exp_scale).sum(dim=-1))

        # rew["dof_force_target"] = square_sum(self.dof_force_target)
        
        # rew["dof_force_target"] = out_of_float_bound_squared_sum(
        #     self.dof_force_target, -self.torque_penalty_bound, self.torque_penalty_bound
        # )
        # rew["dof_force_target"] = out_of_bound_norm(self.dof_force_target, -self.torque_penalty_bound, self.torque_penalty_bound)

        # joint acc penalty
        # rew["dof_acc"] = torch.square(self.last_dof_vel - self.dof_vel).sum(dim=1)
        self.dof_acc = (self.dof_vel - self.last_dof_vel) * self.rl_dt_inv  # TODO check if [:] is needed # TODO
        rew["dof_acc"] = square_sum(self.dof_acc)  # TODO Check this
        # rew["dof_acc"] = torch.abs(dof_acc).sum(dim=1)  #TODO Check this

        # joint vel penalty
        rew["dof_vel"] = square_sum(self.dof_vel)

        # joint position penalty
        # rew["dof_pos"] = (self.dof_pos - self.default_dof_pos).abs().sum(dim=1)
        rew["dof_pos"] = (self.dof_pos - self.desired_dof_pos).abs().sum(dim=1)
        rew["dof_pos"][self.is_zero_command] = 0
        # rew["dof_pos"] = (self.dof_pos_filt - self.desired_dof_pos).abs().sum(dim=1)
        
        # joint power penalty
        rew["dof_pow"] = (self.dof_vel * self.dof_force_target).abs().sum(dim=1)

        # penalty for position exceeding dof limit
        # rew["dof_limit"] = out_of_bound_norm(self.dof_pos, self.dof_lower, self.dof_upper)
        rew["dof_limit"] = out_of_bound_abs_sum(self.dof_pos, self.dof_lower, self.dof_upper)

        # collision penalty
        knee_collision = torch.norm(self.contact_force[:, self.knee_ids, :], dim=2) > 1.0
        rew["collision"] = torch.sum(knee_collision, dim=1, dtype=torch.float)  # sum vs any ?

        # foot impact penalty (num_envs,4,3)
        foot_contact_diff = self.foot_contact_force[:, :, 2] - self.last_foot_contact_force[:, :, 2]
        rew["impact"] = foot_contact_diff.abs().sum(dim=1)
        # rew["impact"] = foot_contact_diff.view(self.num_envs,-1).abs().sum(dim=1)
        # foot_contact_diff = self.foot_contact_force - self.last_foot_contact_force
        # rew["impact"] = torch.norm(foot_contact_diff,dim=2).sum(dim=1)

        # stumbling penalty
        stumble = (torch.norm(self.contact_force[:, self.foot_ids, :2], dim=2) > 5.0) * (
            torch.abs(self.contact_force[:, self.foot_ids, 2]) < 1.0
        )
        rew["stumble"] = torch.sum(stumble, dim=1, dtype=torch.float)

        # foot slip penalty
        rew["slip"] = (self.foot_lin_vel.square().sum(dim=2) * self.foot_contact).sum(dim=1)

        # action penalty
        # rew["action"] = torch.square(self.actions).sum(dim=1)
        rew["action"] = self.actions.abs().sum(dim=1)
        # rew["action"] = self.actions_filt.abs().sum(dim=1)  # filtered

        # action rate penalty
        self.action_rate = (self.actions - self.last_actions) * self.rl_dt_inv
        rew["action_rate"] = torch.square(self.action_rate).sum(dim=1)

        # self.is_zero_command = torch.norm(self.commands[:, :2], dim=1) < self.command_zero_threshold
        # nonzero_command = torch.norm(self.commands[:, :2], dim=1) >= self.command_zero_threshold
        foot_no_contact = ~self.foot_contact

        # air time reward (reward long swing)
        first_contact = (self.air_time > 0.0) * self.foot_contact
        self.air_time += self.rl_dt
        # reward only on first contact with the ground
        # rew["air_time"] = (
        #     torch.sum((self.air_time + self.air_time_offset) * first_contact, dim=1, dtype=torch.float)
        #     * nonzero_command  # no reward for zero command
        # )
        # rew["air_time"] = nonzero_command * first_contact * torch.sum((self.air_time + self.air_time_offset), dim=1, dtype=torch.float)
        
        rew["air_time"] = torch.sum((self.air_time + self.air_time_offset)*first_contact, dim=1, dtype=torch.float)
        rew["air_time"][self.is_zero_command] = 0

        self.air_time *= foot_no_contact  # reset if contact

        # foot stance time reward (reward long stance)
        first_no_contact = (self.stance_time > 0.0) * foot_no_contact
        self.stance_time += self.rl_dt
        # reward only on first leaving the ground
        # rew["stance_time"] = (
        #     torch.sum((self.stance_time + self.stance_time_offset) * first_no_contact, dim=1, dtype=torch.float)
        #     * nonzero_command  # no reward for zero command
        # )
        rew["stance_time"] = torch.sum((self.stance_time + self.stance_time_offset) * first_no_contact, dim=1, dtype=torch.float)
        rew["stance_time"][self.is_zero_command] = 0

        self.stance_time *= self.foot_contact  # reset if no contact

        # # reward contact at zero command
        # rew["contact"] = torch.sum(self.foot_contact, dim=1,dtype=torch.float)* (~nonzero_command)

        # penalize high contact forces
        contact_force_norm = torch.norm(self.contact_force[:, self.foot_ids, :], dim=-1)
        rew["contact_force"] = torch.sum((contact_force_norm - self.max_foot_contact_force).clip(min=0.0), dim=1)


        foot_num_contact = torch.sum(self.foot_contact, dim=1, dtype=torch.int) # TODO
        foot_multi_contact = torch.logical_not(foot_num_contact == self.max_single_contact)
        self.foot_multi_contact_time[foot_multi_contact]+=self.rl_dt
        self.foot_multi_contact_time[~foot_multi_contact] = 0.0
        
        rew["single_contact"] = torch.logical_or(
            self.foot_multi_contact_time <self.foot_multi_contact_grace_period, 
            self.is_zero_command).type(torch.float)

        if self.guided_contact:
            foot_height_rew = torch.clamp_max(torch.sum(self.foot_height + self.foot_height_offset,dim=-1),0.1)
            # foot_height_rew[self.is_zero_command] = - foot_height_rew[self.is_zero_command]
            foot_height_rew[self.is_zero_command] = 0
            rew["foot_height"] = foot_height_rew
            
            rew["should_contact"] = torch.eq(self.foot_contact,self.contact_target).mean(dim=1,dtype=torch.float)

            foot_projected_gravity_xy = square_sum(self.foot_projected_gravity[...,:2]).mean(dim=-1)
            rew["foot_orientation"]=torch.exp(self.rew_foot_orient_exp_scale*foot_projected_gravity_xy)

            # rew["should_contact"][zero_command] = 1 # set contstant for zero command

        # rew["num_contacts"] = self.air_time<=self.foot_multi_contact_grace_period


        # # cosmetic penalty for hip motion
        # rew["hip"] = (self.dof_pos[:, self.dof_hip_ids] - self.desired_dof_pos[:, self.dof_hip_ids]).abs().sum(dim=1)

        # log episode reward sums
        for key in rew.keys():
            self.episode_sums[key] += rew[key]  # unscaled
            rew[key] *= self.rew_scales[key]
        # total reward
        stacked_rewards = torch.stack(list(rew.values()), dim=0)
        self.rew_buf = torch.sum(stacked_rewards, dim=0)
        self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        if self.enable_udp:  # send UDP info to plotjuggler
            data = {
                "t": self.control_steps*self.rl_dt,
                "action": self.actions,
                "action_is_on": self.action_is_on,
                "action_rate": self.action_rate,
                "dof_vel": self.dof_vel,
                "dof_acc": self.dof_acc,
                "dof_pos": self.dof_pos,
                "dof_pos_target": self.dof_pos_target,
                "dof_force_target": self.dof_force_target,
                "dof_force": self.dof_force,
                "base_lin_vel": self.base_lin_vel,
                "base_ang_vel": self.base_ang_vel,
                "base_height": self.base_height,
                "foot_height": self.foot_height,
                "projected_gravity": self.projected_gravity,
                "time_air": self.air_time,
                "time_stance": self.stance_time,
                "foot_pos": self.foot_pos,
                "contact": self.foot_contact,
                "phase": self.phase,
                "should_contact":self.contact_target,
                "rew_buf": self.rew_buf * self.rl_dt_inv,
                "commands": self.commands,
                "rew": {key: rew[key] * self.rl_dt_inv for key in rew},
                "rew_rel":{key: rew[key]/self.rew_buf for key in rew},
                "foot_rb_state": self.rb_state[:,self.foot_ids],
                # "foot_quat": self.foot_quat,
                "base_quat": self.base_quat,
                "root_state": self.root_state,
            }
              
            if self.items_to_publish is not None:
                data = {key: data[key] for key in self.items_to_publish}
            self.data_publisher.publish({"sim":data})  
            # self.sim2real_publisher.publish({
            #     "dof_pos_target": self.dof_pos_target[0],
            # })
    
    @property
    def foot_pos(self):
        """return foot position relative to the base in the body frame"""
        foot_pos_rel = self.rb_state[:, self.foot_ids, 0:3] - self.root_state[:, :3].view(self.num_envs, 1, 3)
        return quat_rotate_inverse(self.base_quat.repeat_interleave(self.num_foot, dim=0), 
                                   foot_pos_rel.view(-1, 3)).view(self.num_envs, self.num_foot, 3)

    def reset_idx(self, env_ids):
        """Resets the specified environments."""
        len_ids = len(env_ids)
        if len_ids == 0:
            return
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env_ids_raw = gymtorch.unwrap_tensor(env_ids_int32)
        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_state[env_ids] = self.base_init_state
            self.root_state[env_ids, :3] += self.env_origins[env_ids]
            self.root_state[env_ids, :2] += torch.empty(len_ids, 2, device=self.device, dtype=torch.float).uniform_(*self.base_pos_xy_range)
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

        # set small commands to zero
        self.is_zero_command[env_ids] = torch.norm(self.commands[env_ids, :3], dim=1) < self.command_zero_threshold
        self.commands[self.is_zero_command]=0

        if self.enable_erfi: 
            self.erfi_rao[env_ids] = torch.empty(len_ids, self.num_dof,dtype=torch.float, device=self.device).uniform_(*self.erfi_rao_range)
    
        # # TODO: reset some observations
        # self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        self.foot_multi_contact_time[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        # self.actions_filt[env_ids] = 0.0
        # self.dof_pos_filt[env_ids] = self.dof_pos[env_ids]
        self.last_foot_contact_force[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.stance_time[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
    
        # # reset delayed observations
        # self.delayed_obs_buf.reset_batch(env_ids)
        
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
        self.terrain_levels[env_ids] -= 1 * (distance < command_distance * 0.25)
        # TODO check level up/down condition
        # self.terrain_levels[env_ids] += 1 * (distance > command_distance*0.5)
        # level up if run over the terrain or move at least 75% of the command distance
        # self.terrain_levels[env_ids] += 1 * torch.logical_or(
        #     distance > self.terrain.env_length / 2, distance > command_distance * 0.75)
        self.terrain_levels[env_ids] += 1 * torch.logical_or(
            distance > self.terrain.env_length / 2, distance > command_distance * 0.8)
        
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain.env_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.terrain_level_mean = self.terrain_levels.float().mean()

    def push_robots(self):
        """Applies random pushes to the robots."""
        self.root_state[:, 7:13] = torch_rand_tensor(
            self.push_vel_min, self.push_vel_max, (self.num_envs, 6), device=self.device
        )  # lin vel x/y/z
        self.gym.set_actor_root_state_tensor(self.sim, self.root_state_raw)

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
                    # reset
                    self.progress_buf[:]= self.max_episode_length
                elif evt.action == "ref_env-" and evt.value > 0:
                    self.ref_env = (self.ref_env-1)%self.num_envs
                elif evt.action == "ref_env+" and evt.value > 0:
                    self.ref_env = (self.ref_env+1)%self.num_envs
            if self.enable_keyboard_operator:
                for evt in events:
                    if evt.action == "vx+" and evt.value > 0:
                        self.keyboard_operator_cmd[0] += 0.1
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vx-" and evt.value > 0:
                        self.keyboard_operator_cmd[0] -= 0.1
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vy+" and evt.value > 0:
                        self.keyboard_operator_cmd[1] += 0.1
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "vy-" and evt.value > 0:
                        self.keyboard_operator_cmd[1] -= 0.1
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "heading+" and evt.value > 0:
                        self.keyboard_operator_cmd[2] += 0.1
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "heading-" and evt.value > 0:
                        self.keyboard_operator_cmd[2] -= 0.1
                        print(f"{self.keyboard_operator_cmd}")
                    elif evt.action == "v=0" and evt.value > 0:
                        self.keyboard_operator_cmd[:] = 0
                        print(f"{self.keyboard_operator_cmd}")
                
                if self.data_receiver_data_id != self.data_receiver.data_id:
                    self.data_receiver_data_id = self.data_receiver.data_id
                    if "cmd" in self.data_receiver.data:
                        self.keyboard_operator_cmd[:] = torch.tensor(self.data_receiver.data["cmd"],device=self.device)
                    print(f"keybaord cmd:{self.keyboard_operator_cmd}")
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
                self.cam_target_pos = self.root_state[self.ref_env, :3].clone().cpu() 
                self.cam_pos = self.viewer_follow_offset.cpu()+self.cam_target_pos
                self.gym.viewer_camera_look_at(
                    self.viewer, self.envs[self.ref_env], gymapi.Vec3(*self.cam_pos), gymapi.Vec3(*self.cam_target_pos))   

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

        self.actions = torch.clamp(actions, -self.clip_actions, self.clip_actions).clone().to(self.device)

        if self.enable_passive_dynamics:
            self.dof_pos_target = self.action_scale * self.actions[:,:self.num_dof] + self.default_dof_pos
            self.action_is_on = self.actions[:,self.num_dof:]  > 0 
        else:
            self.dof_pos_target = self.action_scale * self.actions + self.default_dof_pos
            self.action_is_on = 1
        # self.actions_filt[:] = self.actions_filt * 0.03 + self.actions * 0.97
        # self.dof_pos_target = self.action_scale * self.actions_filt + self.default_dof_pos

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
        """baseline pre_physics_step, PD position control"""
        for i in range(self.decimation):
            torque = torch.clip(
                self.kp * (self.dof_pos_target - self.dof_pos) - self.kd * self.dof_vel,
                -self.dof_force_target_limit,
                self.dof_force_target_limit,
            )
            if self.enable_passive_dynamics:
                torque*=self.action_is_on
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torque))
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_force_target = torque.view(self.dof_force_target.shape)

    def pre_physics_step_erfi(self):
        """pre_physics_step with extended random force injection"""
        self.erfi_rfi.uniform_(*self.erfi_rfi_range).add_(self.erfi_rao)
        for i in range(self.decimation):
            torque = torch.clip(
                # self.kp * (self.dof_pos_target - self.dof_pos) - self.kd * self.dof_vel + self.erfi_rfi[i],
                self.kp * (self.dof_pos_target - self.dof_pos) - self.kd * self.dof_vel + self.erfi_rfi,
                -self.dof_force_target_limit,
                self.dof_force_target_limit,
            ) # TODO maybe check if action exceeds limit and make it a reward
            if self.enable_passive_dynamics:
                torque*=self.action_is_on
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torque))
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_force_target = torque.view(self.dof_force_target.shape)

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        if self.common_step_counter % self.push_interval == 0 and self.should_push_robots:
            self.push_robots()

        # prepare quantities
        # self.base_quat = self.root_state[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        self.foot_quat = self.rb_state[:, self.foot_ids, 3:7] # foot quaternion [num_envs, num_foot, 4] # this is a copy!!
        self.foot_projected_gravity=quat_rotate_inverse(self.foot_quat.view(-1, 4), self.foot_gravity_vec).view(self.num_envs, self.num_foot, 3)

        if self.viewer and self.enable_keyboard_operator:
            self.commands[:, 3] = wrap_to_pi(2 * self.commands[:, 2] + heading)
        else:
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
        
        # set zero command if the magnitude is too small
        self.is_zero_command[:] = torch.norm(self.commands[:, :3], dim=1) < self.command_zero_threshold
        self.commands[self.is_zero_command]=0

        # foot contact
        # self.contact = torch.norm(contact_forces[:, foot_indices, :], dim=2) > 1.
        self.foot_contact_force = self.contact_force[:, self.foot_ids, :]
        self.foot_contact = self.foot_contact_force[:, :, 2] > 1.0  # todo check with norm
        self.foot_lin_vel = self.rb_state[:, self.foot_ids, 7:10]

        if self.guided_contact: # update phase for the contact sequence
            self.update_phase()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_observations()
        self.compute_reward()

        # update last_...
        # self.dof_pos_filt[:] = self.dof_pos_filt * 0.97 + self.dof_pos * 0.03
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_foot_contact_force[:] = self.foot_contact_force[:]

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
        self.base_height = self.heights_relative[:, self.num_height_points+self.base_id]
        self.foot_height = self.heights_relative[:, self.num_height_points+self.foot_ids]

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
def square_sum(input: torch.Tensor,dim: int=-1) -> torch.Tensor:
    return torch.square(input).sum(dim=dim)


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
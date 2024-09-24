import signal
import sys
import os

import isaacgym
import isaacgymenvs

sys.path.append(isaacgymenvs.__path__[0])  # fix isaacgymenvs imports
sys.path.append(os.path.abspath(__file__ + "/../.."))  # fix envs imports



from common.publisher import DataReceiver
from envs.tasks.legged_terrain import LeggedTerrain
from isaacgymenvs.tasks import isaacgym_task_map
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np
from functools import partial
import dearpygui.dearpygui as dpg

isaacgym_task_map["A1Terrain"] = LeggedTerrain
isaacgym_task_map["AnymalTerrain"] = LeggedTerrain
isaacgym_task_map["RobotDog"] = LeggedTerrain
isaacgym_task_map["Biped"] = LeggedTerrain

# using np.arange [star,end), step, round to 15 decimals
OmegaConf.register_new_resolver(
    "arange", lambda start, stop, step: list(np.round(np.arange(start, stop, step), 15)), replace=True
)

# using np.arange [star,end), step, round to 15 decimals
OmegaConf.register_new_resolver(
    "linspace", lambda start, stop, num: list(np.round(np.linspace(start, stop, num), 15)), replace=True
)


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    # print(OmegaConf.to_yaml(cfg))
    env = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg
    )

    def slider_callback(sender, app_data, user_data):
        """Callback function executed when the slider value changes."""
        id, prop_name, env, format = user_data
        env.dof_props[prop_name][id] = app_data
        env.gym.set_actor_dof_properties(
            env.envs[0], env.actor_handles[0], env.dof_props)
        # print(f"{prop_name}:{app_data:{format}}")  # Print the new slider value to the console

    dpg.create_context()
    dpg.set_global_font_scale(1)
    dpg.create_viewport(title='motor dynamics parameters', width=700, height=450)
    dpg.setup_dearpygui()
    # Disable VSync (Vertical Synchronization)
    dpg.set_viewport_vsync(False)

    def add_slider(id, prop_name, label, min_value=-1, max_value=1.0, format='.3f', env=env):
        dpg.add_drag_float(
            speed=0.001,
            label=label,
            default_value=env.dof_props[prop_name][id],
            min_value=min_value,
            max_value=max_value,
            format=f"%{format}",
            callback=slider_callback,
            width=200,
            # Pass both prop_name and env as a tuple
            user_data=(id, prop_name, env, format)
        )

    with dpg.window(label="", width=700, height=450):

        with dpg.table(header_row=True):
            # Create columns for the grid
            dpg.add_table_column(label="id", init_width_or_weight=0.1)
            dpg.add_table_column(label="    damping")
            dpg.add_table_column(label="    friction")
            dpg.add_table_column(label="    armature")
            for k in range(env.num_dof):  # TODO change back
                with dpg.table_row():
                    dpg.add_text(f"{k}")
                    add_slider(id=k, prop_name="damping", label="",
                               min_value=-5.0, max_value=1.0)
                    add_slider(id=k, prop_name="friction", label="",
                               min_value=-1.0, max_value=1, format='.5f')
                    add_slider(id=k, prop_name="armature", label="",
                               min_value=-1.0, max_value=1)

    dpg.show_viewport()

    # cfg.task.env.
    # Create a receiver instance
    receiver = DataReceiver(port=9871, decoding="msgpack",broadcast=True)
    # Start continuous receiving in a thread
    receiver.receive_continuously()
    received_id = 0

    env.reset()

    default_dof_pos = env.default_dof_pos
    actions = torch.zeros(env.num_envs,env.num_actions, device=env.device)
    action_scale = env.action_scale
    # for k in range(1000000):

    while (not env.gym.query_viewer_has_closed(env.viewer)) and dpg.is_dearpygui_running():
        if receiver.data is not None:
            if receiver.data_id != received_id:
                received_id = receiver.data_id
                # print(f"Received from {receiver.address}: {receiver.data}")
                if "reset" in receiver.data and receiver.data['reset']:
                    env.reset_idx(torch.arange(
                        env.num_envs, device=env.device))
                if "dof_pos_target" in receiver.data:
                    dof_pos_target = torch.tensor(
                        receiver.data["dof_pos_target"], dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
                    actions = (dof_pos_target - default_dof_pos)/action_scale
                if "action_is_on" in receiver.data:
                    action_is_on = torch.tensor(receiver.data["action_is_on"], dtype=torch.float, device=env.device)
                    converted_action_is_on = torch.where(action_is_on>0, 1, -1).repeat(env.num_envs, 1)
                    actions = torch.column_stack((actions, converted_action_is_on))
                if "kp" in receiver.data:
                    env.kp[:] = torch.tensor(receiver.data["kp"], dtype=torch.float, device=env.device)
                if "kd" in receiver.data:
                    env.kd[:] = torch.tensor(receiver.data["kd"], dtype=torch.float, device=env.device)
                
        env.step(actions)
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    print("done")


if __name__ == "__main__":
    launch_rlg_hydra()

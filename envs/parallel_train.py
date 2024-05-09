import argparse
import os
import subprocess
import sys
import time
import threading
import collections



class Launcher:

    def __init__(self,**args) -> None:
        self.max_parallel = args["max_parallel"]
        self.num_gpus=args["num_gpus"]
        self.experiments_per_gpu=args["experiments_per_gpu"]
        self.train_dir=args["train_dir"]
        self.pause_between=args["pause_between"]
        self.log_interval = 3  # seconds
        self.processes = []
        self.processes_per_gpu = {g: [] for g in range(self.num_gpus)}
        self.failed_processes = []
        self.last_log_time = 0
        self.remaining_experiments = collections.deque()

    def add(self,experiments):
        self.remaining_experiments.extend(experiments)
        
    def run(self):
        # self.thread.start()
        self._run()
    
    def find_least_busy_gpu(self):
        least_busy_gpu = None
        gpu_available_processes = 0

        for gpu_id in range(self.num_gpus):
            available_processes = self.experiments_per_gpu - len(self.processes_per_gpu[gpu_id])
            if available_processes > gpu_available_processes:
                gpu_available_processes = available_processes
                least_busy_gpu = gpu_id

        return least_busy_gpu, gpu_available_processes

    def can_squeeze_another_process(self):
        if len(self.processes) >= self.max_parallel:
            return False
        if self.experiments_per_gpu > 0:
            least_busy_gpu, gpu_available_processes = self.find_least_busy_gpu()
            if gpu_available_processes <= 0:
                return False
        return True
    
    def _run(self):
        # print("Starting processes with base cmds: %r", [e.cmd for e in experiments])
        # print(f"Max parallel processes is {self.max_parallel}")
        # print(f"Monitor log files using\n\n\ttail -f train_dir/{run_description.run_name}/**/**/sf_log.txt\n\n")

        if len(self.remaining_experiments)>0:
            next_experiment = self.remaining_experiments.pop()
        else:
            next_experiment=None
        
        while len(self.processes) > 0 or next_experiment is not None:
            while self.can_squeeze_another_process() and next_experiment is not None:
                
                cmd = next_experiment["cmd"]
                root_dir=next_experiment["root_dir"]
                exp_env_vars=next_experiment["exp_env_vars"]

                cmd_tokens = cmd.split(" ")

                # workaround to make sure we're running the correct python executable from our virtual env
                if cmd_tokens[0].startswith("python"):
                    cmd_tokens[0] = sys.executable
                    # print(f"Using Python executable {cmd_tokens[0]}")

                expeirment_path=os.path.join(self.train_dir, root_dir)
                if not os.path.exists(expeirment_path):
                    os.makedirs(expeirment_path, exist_ok=True)
                    # os.chdir(expeirment_path)
                    # os.chdir("/home/grl/repo/RobotsMakingRobots/legged_env/envs")

                envvars = os.environ.copy()

                best_gpu = None
                if self.experiments_per_gpu > 0:
                    best_gpu, best_gpu_available_processes = self.find_least_busy_gpu()
                    # print(f"The least busy gpu is {best_gpu} where we can run {best_gpu_available_processes} more processes")
                    envvars["CUDA_VISIBLE_DEVICES"] = f"{best_gpu}"

                # print(f"Starting process {cmd}")

                if exp_env_vars is not None:
                    for key, value in exp_env_vars.items():
                        print(f"Adding env variable {key} {value}")
                        envvars[str(key)] = str(value)

                process = subprocess.Popen(cmd_tokens, stdout=None, stderr=None,env=envvars)
                
                process.gpu_id = best_gpu
                process.proc_cmd = cmd

                self.processes.append(process)

                if process.gpu_id is not None:
                    self.processes_per_gpu[process.gpu_id].append(process.proc_cmd)

                print(f"Started process {process.proc_cmd} GPU {process.gpu_id}")
                # print(f"Waiting for {self.pause_between} seconds before starting next process")
                time.sleep(self.pause_between)

                if len(self.remaining_experiments)>0:
                    next_experiment = self.remaining_experiments.pop()
                else:
                    next_experiment=None


            remaining_processes = []
            for process in self.processes:
                if process.poll() is None:
                    remaining_processes.append(process)
                    continue
                else:
                    if process.gpu_id is not None:
                        self.processes_per_gpu[process.gpu_id].remove(process.proc_cmd)
                    print(f"Process finished {process.proc_cmd}, {process.returncode}")
                    if process.returncode != 0:
                        self.failed_processes.append((process.proc_cmd, process.pid, process.returncode))
                        print(f"WARNING: RETURN CODE IS {process.returncode}")

            self.processes = remaining_processes

            if time.time() - self.last_log_time > self.log_interval:
                if self.failed_processes:
                    print(f"Failed processes:", ", ".join([f"PID: {p[1]} code: {p[2]}" for p in self.failed_processes]))
                self.last_log_time = time.time()
            time.sleep(0.1)
            

        print("Done!")

        return 0


if __name__=="__main__":
    
    # entry_point="envs/example_entry_point.py"
    # entry_point=os.path.abspath(entry_point)
    # experiments = [dict(cmd=f"python {entry_point}",root_dir=f"exp_{k}",exp_env_vars=None) for k in range(5)]


    # cmd_lists = ["python train.py task=A1Terrain test=true num_envs=2 task.env.terrain.terrainType=plane"]
    cmd_lists = ["python train.py task=RobotDog test=True num_envs=2 task.env.urdfAsset.file='scaled_top_legs_dog.urdf' ++task.env.urdfAsset.root='/home/grl/repo/RobotsMakingRobots/assets/URDFsForBoxi/scaled_top_legs_dog' task.env.baseHeightTarget=0.2694601156286657 task.env.baseInitState.pos=[0,0,0.2694601156286657] task.env.baseInitState.rot=[0,0,0,1] task.env.defaultJointAngles.joint_1_0=0 task.env.defaultJointAngles.joint_1_1=0 task.env.defaultJointAngles.joint_1_2=0 task.env.defaultJointAngles.joint_1_3=0 task.env.defaultJointAngles.joint_2_0=0 task.env.defaultJointAngles.joint_2_1=0 task.env.defaultJointAngles.joint_2_2=0 task.env.defaultJointAngles.joint_2_3=0 task.env.desiredJointAngles.joint_1_0=0 task.env.desiredJointAngles.joint_1_1=0 task.env.desiredJointAngles.joint_1_2=0 task.env.desiredJointAngles.joint_1_3=0 task.env.desiredJointAngles.joint_2_0=0 task.env.desiredJointAngles.joint_2_1=0 task.env.desiredJointAngles.joint_2_2=0 task.env.desiredJointAngles.joint_2_3=0 task.env.urdfAsset.hipName=body_1 task.env.urdfAsset.kneeName=body_1 task.env.urdfAsset.footName=body_2 task.env.urdfAsset.hipJointName=joint_1 task.env.terrain.terrainType=plane"]
    
    
    
    experiments = [dict(cmd=cmd,root_dir=f"exp_{k}",exp_env_vars=None) for k,cmd in enumerate(cmd_lists)]

    
    launcher = Launcher(max_parallel=4,num_gpus=1,experiments_per_gpu=2,train_dir="./output/tmp", pause_between=0)
    launcher.add(experiments)
    launcher.run()
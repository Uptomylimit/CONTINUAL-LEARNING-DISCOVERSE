import time
import collections
import dm_env
import importlib
from discoverse.envs.airbot_play_base import AirbotPlayCfg
from discoverse.task_base import AirbotPlayTaskBase

import gymnasium as gym
import numpy as np
from envs.common_env import get_image_numpy, CommonEnv
import torch as th
import os
import pickle


class MujocoEnv(gym.Env):
    """
    Mujoco environment for airbot_play
    path: path to the script containing the SimNode class and config (mainly for data collection)
    """

    def __init__(self, path: str,env_config):
        self.env_config = env_config
        self.policy_class = env_config["policy_class"]

        module = importlib.import_module(path.replace("/", ".").replace(".py", ""))
        node_cls = getattr(module, "SimNode")
        cfg: AirbotPlayCfg = getattr(module, "cfg")
        cfg.headless = True
        # cfg.headless = False
        self.exec_node: AirbotPlayTaskBase = node_cls(cfg)
        # self.exec_node.cam_id = self.exec_node.config.obs_camera_id
        self.reset_position = None
        print("MujocoEnv initialized")

        # add qpos and action normalization to 0 1
        # add action clip
        self.qpos_mean = [0]*self.env_config["action_dim"]
        self.qpos_std = [1]*self.env_config["action_dim"]
        self.action_mean = [0]*self.env_config["action_dim"]
        self.action_std = [1]*self.env_config["action_dim"]


        if self.env_config["stage"]=="train":
            print("train load stats!!!!!")
            stats_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                "policies/" + self.env_config['load_policy'],
                                "my_ckpt",
                                self.env_config['task_name'],
                                self.env_config['load_time_stamp'],
                                self.env_config['task_name'],
                                self.env_config['load_time_stamp'],
                                f'dataset_stats.pkl')
        else:
            print("eval load stats!!!!!")
            stats_path = os.path.join(os.path.dirname(os.path.dirname(self.env_config["ckpt_dir"])),
                                self.env_config['load_time_stamp'],
                                "ckpt",
                                  f'best_modelmeanstd.pkl')

        # stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl') if stats_path == '' else stats_path
        # print(f'Saving stats into {stats_path}...')
        # with open(stats_path, 'wb') as f:
        #     pickle.dump(stats, f)

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
            print(f'Env: Loading stats from {stats_path}...')
        # 在env环境中对qpos和action进行标准化
        self.qpos_mean = stats["qpos_mean"]
        self.qpos_std = stats["qpos_std"]
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]

        # print("------------------  stats  ------------------")
        # print(self.qpos_mean,self.qpos_std,self.action_mean,self.action_std)


        self.action_max = [2.09, 0.17, 3.14, 2.96, 1.74, 3.14, 1]
        self.action_min = [-3.14, -2.96, -0.087, -2.96, -1.74, -3.14, 0]
        self.max_steps = self.env_config["max_timesteps"]
        self.steps = 0

        image_space = gym.spaces.Box(
            low=0, high=1, shape=(
                len(env_config["camera_names"]),3,self.exec_node.height, self.exec_node.width), dtype=np.float32)
        qpos_space = gym.spaces.Box(
            low=np.array(self.action_min), high=np.array(self.action_max), shape=(7,), dtype=np.float32)
        
        # 观测空间为字典，包含图像和关节角度
        self.observation_space = gym.spaces.Dict({
            'images': image_space,
            'qpos': qpos_space
        })
        
        # 动作空间：25×7的关节角度矩阵
        if self.env_config["policy_config"]["temporal_agg"]:
            # 为了使得rolloutbuffer读取到的action的维度是没有chunksize的
            self.action_space = gym.spaces.Box(low=np.array(self.action_min), 
                                           high=np.array(self.action_max), 
                                           shape=(env_config["action_dim"],), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=np.array([self.action_min for _ in range(env_config["policy_config"]["chunk_size"])]).flatten(), 
                                           high=np.array([self.action_max for _ in range(env_config["policy_config"]["chunk_size"])]).flatten(), 
                                           shape=(env_config["action_dim"]*env_config["policy_config"]["chunk_size"],), dtype=np.float32)


    def set_reset_position(self, reset_position):
        self.reset_position = reset_position
        print("Resetting to the given position: ", self.reset_position)

    def get_reward(self):
        return self.exec_node.getReward()
    
        # if self.exec_node.check_success():
        #     return 1
        # return 0

    def pre_process(self,a , b, c):
        # 确保输入的列表长度相同
        if len(a) != len(b) or len(a) != len(c):
            raise ValueError("输入的列表长度必须相同")
        
        # # 计算结果
        # for i in range(len(a)):
        #     a[i] = (a[i] - b[i]) / c[i]

        return (a-b)/c
    
    def post_process(self,a, b, c):
        # 确保输入的列表长度相同
        if len(a) != len(b) or len(a) != len(c):
            # print(a,b,c)
            raise ValueError("输入的列表长度必须相同")
        
        if self.policy_class == "ACT":
            # for i in range(len(a)):
            #     a[i] = a[i]*c[i] + b[i]
            return a*c + b
        elif self.policy_class == "Diffusion":
            return ((a + 1) / 2) * (self.action_max - self.action_min) + self.action_min
            



    def reset(self, seed=0):
        # print("------------------")
        # print(self.action_mean,self.action_std)
        self.steps = 0

        self.exec_node.domain_randomization() # 合并到self.exec_node.reset()中
        raw_obs = self.exec_node.reset()
        
        # if self.reset_position is not None:
        #     # self.reset_position[-1] = 0.96
        #     # print("Resetting to the given position: ", self.reset_position)
        #     self.reset_position[-1] = 0.04  # undo the normalization
        #     self.exec_node.mj_data.ctrl[:7] = self.reset_position
        #     self.exec_node.mj_data.ctrl[7] = -self.exec_node.mj_data.ctrl[6]
        #     self.exec_node.mj_data.qpos[:7] = self.reset_position
        #     self.exec_node.mj_data.qpos[7] = -self.exec_node.mj_data.qpos[6]
        #     raw_obs, pri_obs, rew, ter, info = self.exec_node.step(self.reset_position)
        # time.sleep(sleep_time)
        obs = collections.OrderedDict()
        obs["qpos"] = np.array(raw_obs["jq"])
        # print("pre_obs", obs["qpos"])

        # self.pre_process(obs["qpos"],self.qpos_mean,self.qpos_std)
        obs['qpos'] = self.pre_process(obs["qpos"],self.qpos_mean,self.qpos_std) # normalization

        # print("obs gripper", raw_obs["jq"][-1])
        # print("pre_obs", obs["qpos"])
        # obs["qpos"][-1] *= 25  # undo the normalization
        # print("post_obs", obs["qpos"])
        # obs["qvel"] = raw_obs["jv"]
        obs["images"] = {}
        for id in self.exec_node.config.obs_rgb_cam_id:
            obs["images"][f"{id}"] = raw_obs["img"][id][:, :, ::-1]

        camera_names = self.env_config["camera_names"]
        ts = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs,)
        curr_image = get_image_numpy(ts, camera_names, mode=0)

        obs["images"] = curr_image

        return obs,{"TimeLimit.truncated":False,
                    "terminal_observation":obs}
        # return dm_env.TimeStep(
        #     step_type=dm_env.StepType.FIRST,
        #     reward=self.get_reward(),
        #     discount=None,
        #     observation=obs,
        # )

    def step(
        self,
        action,
        get_obs=True,
        sleep_time=0,
    ):

        if not self.env_config["policy_config"]['temporal_agg']:

            self.steps += 10
        
            action = action.reshape(self.env_config["policy_config"]["chunk_size"],self.env_config["action_dim"])
            for i in range(self.env_config["policy_config"]["chunk_size"]):
                # print(action[i],self.action_mean,self.action_std)

                action[i] = self.post_process(action[i],self.action_mean,self.action_std) # denormalization

                raw_obs, pri_obs, rew, ter, info = self.exec_node.step(action[i])

            time.sleep(sleep_time)

            if get_obs:
                obs = collections.OrderedDict()
                # obs["qpos"] = list(raw_obs["jq"])
                obs["qpos"] = np.array(raw_obs["jq"])
                obs['qpos'] = self.pre_process(obs["qpos"],self.qpos_mean,self.qpos_std) # normalization
                obs["images"] = {}
                for id in self.exec_node.config.obs_rgb_cam_id:
                    obs["images"][f"{id}"] = raw_obs["img"][id][:, :, ::-1]
            else:
                obs = None

            camera_names = self.env_config["camera_names"]
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=self.get_reward(),
                discount=None,
                observation=obs,)
            curr_image = get_image_numpy(ts, camera_names, mode=0)

            obs["images"] = curr_image

            terminated = True if self.steps>=self.max_steps else False

            return obs,self.get_reward(),terminated,False,{"TimeLimit.truncated":True,
                                                            "terminal_observation":obs}
        
        else:
            # print("temporal_agg",self.env_config["policy_config"]["temporal_agg"])

            self.steps += 1

            # print("action",action.shape)

            action = action.reshape(self.env_config["policy_config"]["chunk_size"],self.env_config["action_dim"])

            action = self.post_process(action[0],self.action_mean,self.action_std) # denormalization

            raw_obs, pri_obs, rew, ter, info = self.exec_node.step(action)

            time.sleep(sleep_time)

            if get_obs:
                obs = collections.OrderedDict()
                # obs["qpos"] = list(raw_obs["jq"])
                obs["qpos"] = np.array(raw_obs["jq"])
                obs['qpos'] = self.pre_process(obs["qpos"],self.qpos_mean,self.qpos_std) # normalization
                obs["images"] = {}
                for id in self.exec_node.config.obs_rgb_cam_id:
                    obs["images"][f"{id}"] = raw_obs["img"][id][:, :, ::-1]
            else:
                obs = None

            camera_names = self.env_config["camera_names"]
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=self.get_reward(),
                discount=None,
                observation=obs,)
            curr_image = get_image_numpy(ts, camera_names, mode=0)

            obs["images"] = curr_image

            truncated = True if self.steps>=self.max_steps else False

            return obs,self.get_reward(),False,truncated,{"TimeLimit.truncated":truncated,
                                                            "terminal_observation":obs}


        # return dm_env.TimeStep(
        #     step_type=dm_env.StepType.MID,
        #     reward=self.get_reward(),
        #     discount=None,
        #     observation=obs,
        # )


def make_env(path,env_config):
    env = MujocoEnv(path,env_config)
    return env

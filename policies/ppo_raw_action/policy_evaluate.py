import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import shutil
import time
import argparse

from utils.utils import load_data, LoadDataConfig, compute_dict_mean, set_seed, detach_dict, GPUer
from configurations.task_configs.config_tools.basic_configer import basic_parser, get_all_config
from policies.common.maker import make_policy
import logging

from policies.ppo.ppo import PPO
from policies.ppo.common.policies import ACT_ActorCriticPolicy
from envs.common_env import get_image, CommonEnv
from policies.ppo.common.env_util import make_vec_env
from policies.ppo.common.evaluation import evaluate_policy
from policies.ppo.common.callbacks import EvalCallback
import cv2
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args:dict):
    # get all config
    all_config = get_all_config(args, 'train')
    all_config["stage"] = "eval"
    print("-------------------")
    # print(all_config)
    ckpt_dir = all_config['ckpt_dir']
    print("ckpt_dir",ckpt_dir)
    stats_path = all_config['stats_path']
    gpu_threshold = all_config.get('gpu_threshold', 10)

    # print(all_config)

    set_seed(all_config["seed"])
    ckpt_names = all_config["ckpt_names"]

    # make environment
    env_config = all_config["environments"]
    env_maker = env_config.pop("environment_maker")
    env = env_maker(all_config)  # use all_config for more flexibility
    assert env is not None, "Environment is not created..."

    # env.reset()
    # print("get_reward",env.get_reward())


    # # 加载数据及统计信息
    # train_dataloader, val_dataloader, stats = load_data(
    #     LoadDataConfig(
    #         **all_config["load_data"],
    #         camera_names=all_config["camera_names"],
    #         chunk_sizes={"action": all_config["policy_config"]["chunk_size"]},
    #     )
    # )

    # 导入act保存统计数据
    # print(all_config['task_name'])
    # print(all_config['load_time_stamp'])
    # print(all_config['load_policy'])
    # print(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
    #                           "policies/" + all_config['load_policy'],
    #                           "my_ckpt",
    #                           all_config['task_name'],
    #                           all_config['load_time_stamp'],
    #                           all_config['task_name'],
    #                           all_config['load_time_stamp'],
    #                           f'dataset_stats.pkl'))


    # wait for free GPU
    target_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    target_gpus = target_gpus.split(',')
    target_gpus = [int(gpu) for gpu in target_gpus if gpu != '']
    waiting_time = 60
    while True:
        free_gpus, gpus_num, gpu_utilizations = GPUer.check_all_gpus_idle(gpu_threshold)
        if len(free_gpus) > 0:
            if len(target_gpus) > 0:
                not_free_ids = []
                for index, target_gpu in enumerate(target_gpus):
                    if target_gpu >= gpus_num:
                        raise ValueError(f'Target GPU id (from 0) {target_gpu} is not valid, only {gpus_num} gpus available')
                    elif target_gpu not in free_gpus:
                        not_free_ids.append(index)
                if len(not_free_ids) != 0:
                    print(f'Target GPU {target_gpus[not_free_ids]} is not free ({gpu_utilizations[not_free_ids]}), waiting for {waiting_time} senconds...')
                    time.sleep(waiting_time)
                    continue
            else:
                free_gpu = free_gpus[0]
                os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
                print(f'Using GPU {free_gpu}')
                target_gpus = [int(free_gpu)]
            break
        else:
            print(f'No free GPU, waiting for {waiting_time} senconds...')
            time.sleep(waiting_time)
    print(f"Using GPU: {target_gpus}")
    all_config["target_gpus"] = target_gpus


    # train policy
    set_seed(all_config['seed'])
    eval_info = eval_bc(env,env_maker, all_config)
    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def eval_bc(env,env_maker, config):
    # num_epochs = config['num_epochs']
    # ckpt_dir = config['ckpt_dir']
    # seed = config['seed']
    policy_config = config['policy_config']
    # stats_dir = os.path.dirname(config['stats_path'])
    # eval_every = 3.14 if config["eval_every"] == 0 else config["eval_every"]
    parallel = config["parallel"]

    # make policy and load trained policy of act
    # pretrain_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
    #                         "policies/ppo",
    #                         "my_ckpt/ckpt/best_model",
    #                         "policy.pth")
    
    pretrain_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                            "policies/ppo",
                            "my_ckpt/ckpt/2025.2.24/step25000")
    pretrain_name = "step25000.ckpt"
    
    # pretrain_path = "/home/sfw/project/pythonproject/CONTINUAL-LEARNING-DISCOVERSE/policies/act/my_ckpt/state4_rl/20250218-220243/state4_rl/20250218-220243/policy_best.ckpt"
    # policy_config["ckpt_path"] = pretrain_path
    policy_config["ckpt_path"] = os.path.join(pretrain_dir,pretrain_name)
    # policy_config["ckpt_path"] = ""
    # print(f'Loading pretrained policy from {pretrain_path}...')
    # policy = make_policy(policy_config, "train")


    # set GPU device
    if parallel is not None:
        if parallel["mode"] == "DP":
            if parallel.get("device_ids", None) is not None:
                device_ids = parallel["device_ids"]
            else:
                device_ids = config["target_gpus"]
            assert len(device_ids) > 1, "DP mode requires more than 1 GPU"
            print(f'Using GPUs {device_ids} for DataParallel training')
            device_ids = list(range(len(device_ids)))
            policy = torch.nn.DataParallel(policy, device_ids=device_ids)
            optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)
        elif parallel["mode"] == "DDP":
            # TODO: can not use DDP for now
            raise NotImplementedError
            policy = torch.nn.parallel.DistributedDataParallel(policy, device_ids=parallel["device_ids"])
        else:
            raise ValueError(f'Invalid parallel mode: {parallel["mode"]}')
        
    policy_config["stage"] = "train"
    policy_config["ppo_eval"] = True

    # train_env = make_vec_env(env_maker,config, n_envs=1)  # 訓練環境
    eval_env = make_vec_env(env_maker,config, n_envs=1)   # 驗證環境
    ppo = PPO(ACT_ActorCriticPolicy,eval_env,policy_config,n_steps=20,batch_size=10)
    # eval_callback = EvalCallback(eval_env, best_model_save_path=config['ckpt_dir'],
    #                          log_path=config['ckpt_dir'], eval_freq=40,
    #                          n_eval_episodes=5, deterministic=True, render=False)
    # ppo.learn(total_timesteps=1000,callback=eval_callback)

    # ppo.load(path=os.path.join(os.path.dirname(os.path.dirname(config["ckpt_dir"])),
    #                             config['load_time_stamp'],
    #                             "ckpt/best_model"),
    #         env=eval_env,
    #         config=policy_config)
    
    # eval_result_path = os.path.join(os.getcwd(),
    #                         "eval_results",
    #                         config['task_name'],
    #                         config['load_time_stamp'],
    #                         datetime.now().strftime("%Y%m%d-%H%M%S"),
    #                         )
    # if not os.path.exists(eval_result_path):
    #     os.makedirs(eval_result_path)
    # print(f'Saving results into {eval_result_path}...')

    import cv2
    reward_episode = []
    for i in range(config["eval_episodes"]):
        image_list = []
        dt = 1 / config["fps"]
        obs,_ = env.reset()
        rewards = 0
        array = (obs["images"] * 255).astype(np.uint8)
        tran_array = []
        for j in range(len(array)):
            tran_array.append(np.moveaxis(array[j], 0, -1))
            con_image = np.hstack(tran_array)
            image_list.append(con_image)
        for _ in range(config["max_timesteps"]):
            # action, _states = ppo.predict(obs,deterministic=True)
            action = ppo.policy.eval_predict(obs)
            # print("action: ",action)
            obs, reward, _, dones, info = env.step(action)
            # print("reward: ",reward)
            # print(obs["images"].shape)
            rewards += reward
            array = (obs["images"] * 255).astype(np.uint8)
            tran_array = []
            for j in range(len(array)):
                tran_array.append(np.moveaxis(array[j], 0, -1))
            con_image = np.hstack(tran_array)
            # print("con_image.shape",con_image.shape)
            # print(con_image)

            image_list.append(con_image)
            if dones:
                ppo.policy.policy_net.eval_temporal_ensembler.reset()
                break

        # 视频保存参数
        fps = config["fps"]
        # fps = 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式

        # print("save_dir",config["save_dir"])
        # 保存拼接后的图像为视频
        out = cv2.VideoWriter(pretrain_dir + f'/result_policy_best_{i}.mp4', fourcc, fps, (con_image.shape[1], con_image.shape[0]))
        for image in image_list:
            # print(image.shape)
            out.write(image)
        out.release()

        reward_episode.append(rewards)

    env.close()
    print(reward_episode)
    print("mean_reward:",np.mean(reward_episode))



    # from visualize_episodes import save_videos
    # save_path = config["save_dir"]
    # print(save_path)
    # save_videos(image_list, dt, video_path=f"{save_path}.mp4", decompress=False)

    # # test policy forward
    # ts = env.reset()
    # # pre-process current observations
    # camera_names = config["camera_names"]
    # curr_image = get_image(ts, camera_names, mode=0)
    # qpos_numpy = np.array(ts.observation["qpos"])
    # qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)

    # obs = {
    #     'qpos': qpos,  # 用于存储关节角度数据
    #     'images': curr_image  # 用于存储图像数据
    # }

    # out = ppo.policy(obs)
    # print("out:",out)


def parser_eval(parser:argparse.ArgumentParser = None):
    if parser is None:
        parser = basic_parser()
    parser.add_argument('-bs', '--batch_size', action='store', type=int, help='batch_size', required=False)
    parser.add_argument('-lr', '--learning_rate', action='store', type=float, help='learning_rate', required=False)
    parser.add_argument('-ne', '--num_epochs', action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('-pcp','--pretrain_ckpt_path', action='store', type=str, help='pretrain_ckpt_path', required=False)
    parser.add_argument('-peb','--pretrain_epoch_base', action='store', type=str, help='pretrain_epoch_base', required=False)
    parser.add_argument('-ee', '--eval_every', action='store', type=int, help='eval_every')
    parser.add_argument('-ve', '--validate_every', action='store', type=int, help='validate_every', required=False)
    parser.add_argument('-se', '--save_every', action='store', type=int, help='save_every', required=False)
    parser.add_argument('-smd','--skip_mirrored_data', action='store', type=bool, help='skip_mirrored_data', required=False)
    parser.add_argument('-gth', '--gpu_threshold', action='store', type=float, help='gpu_threshold', required=False)
    # set time_stamp  # TODO: used to load pretrain model
    parser.add_argument("-ts", "--time_stamp", action="store", type=str, help="time_stamp", required=False)
    # parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    # parser.add_argument('--history_len', action='store', type=int)
    # parser.add_argument('--future_len', action='store', type=int)
    # parser.add_argument('--prediction_len', action='store', type=int)

    # ppo add 
    parser.add_argument(
        "-can",
        "--can_buses",
        action="store",
        nargs="+",
        type=str,
        help="can_bus",
        default=("can0", "can1"),
        required=False,
    )
    parser.add_argument(
        "-em",
        "--eef_mode",
        action="store",
        nargs="+",
        type=str,
        help="eef_mode",
        default=("gripper", "gripper"),
    )
    parser.add_argument(
        "-bat",
        "--bigarm_type",
        action="store",
        nargs="+",
        type=str,
        help="bigarm_type",
        default=("OD", "OD"),
    )
    parser.add_argument(
        "-fat",
        "--forearm_type",
        action="store",
        nargs="+",
        type=str,
        help="forearm_type",
        default=("DM", "DM"),
    )
    parser.add_argument("-cki", "--check_images", action="store_true")
    parser.add_argument(
        "-rn",
        "--robot_name",
        action="store",
        type=str,
        help="robot_name",
        required=False,
    )
    parser.add_argument( # 有啥用？
        "-ci",
        "--camera_indices",
        action="store",
        nargs="+",
        type=str,
        help="camera_indices",
        default=("0",),
    )
    # set load policy time_stamp
    parser.add_argument(
        "--ckpt_dir",
        action="store",
        type=str,
        help="ckpt_dir",
        required=False,
    )
    parser.add_argument(
        "-lts",
        "--load_time_stamp",
        action="store",
        type=str,
        help="load_time_stamp",
        required=False,
    )

    return parser


if __name__ == '__main__':
    """
    参数优先级：命令行 > config文件
    """
    parser = basic_parser()
    parser_eval(parser)
    main(vars(parser.parse_args()))






































# from habitats.common.robot_devices.cameras.utils import prepare_cv2_imshow

# prepare_cv2_imshow()

# import torch
# import numpy as np
# import os, time, logging, pickle, inspect
# from typing import Dict
# from tqdm import tqdm
# from utils.utils import set_seed, save_eval_results
# from configurations.task_configs.config_tools.basic_configer import (
#     basic_parser,
#     get_all_config,
# )
# from policies.common.maker import make_policy
# from envs.common_env import get_image, CommonEnv
# import dm_env
# import cv2
# import argparse


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def main(args):

#     all_config = get_all_config(args, "eval")
#     set_seed(all_config["seed"])
#     ckpt_names = all_config["ckpt_names"]

#     # make environment
#     env_config = all_config["environments"]
#     env_maker = env_config.pop("environment_maker")
#     env = env_maker(all_config)  # use all_config for more flexibility
#     assert env is not None, "Environment is not created..."

#     results = []
#     # multiple ckpt evaluation
#     for ckpt_name in ckpt_names:
#         success_rate, avg_return = eval_bc(all_config, ckpt_name, env)
#         results.append([ckpt_name, success_rate, avg_return])

#     for ckpt_name, success_rate, avg_return in results:
#         logger.info(f"{ckpt_name}: {success_rate=} {avg_return=}")

#     print()


# def get_ckpt_path(ckpt_dir, ckpt_name, stats_path):
#     ckpt_path = os.path.join(ckpt_dir, ckpt_name)
#     raw_ckpt_path = ckpt_path
#     if not os.path.exists(ckpt_path):
#         ckpt_dir = os.path.dirname(ckpt_dir)  # check the upper dir
#         ckpt_path = os.path.join(ckpt_dir, ckpt_name)
#         logger.warning(
#             f"Warning: not found ckpt_path: {raw_ckpt_path}, try {ckpt_path}..."
#         )
#         if not os.path.exists(ckpt_path):
#             ckpt_dir = os.path.dirname(stats_path)
#             ckpt_path = os.path.join(ckpt_dir, ckpt_name)
#             logger.warning(
#                 f"Warning: also not found ckpt_path: {ckpt_path}, try {ckpt_path}..."
#             )
#     return ckpt_path


# def eval_bc(config, ckpt_name, env: CommonEnv):
#     # TODO: eval only contains the logic, data flow and visualization
#     # remove other not general processing code outside in policy and env maker
#     # 显式获得配置
#     ckpt_dir = config["ckpt_dir"]
#     stats_path = config["stats_path"]
#     save_dir = config["save_dir"]
#     max_timesteps = config["max_timesteps"]
#     camera_names = config["camera_names"]
#     max_rollouts = config["num_rollouts"]
#     policy_config: dict = config["policy_config"]
#     state_dim = policy_config["state_dim"]
#     action_dim = policy_config["action_dim"]
#     temporal_agg = policy_config["temporal_agg"]
#     num_queries = policy_config["num_queries"]  # i.e. chunk_size
#     dt = 1 / config["fps"]
#     image_mode = config.get("image_mode", 0)
#     save_all = config.get("save_all", False)
#     save_time_actions = config.get("save_time_actions", False)
#     filter_type = config.get("filter", None)
#     ensemble: dict = config.get("ensemble", None)
#     save_dir = save_dir if save_dir != "AUTO" else ckpt_dir
#     result_prefix = "result_" + ckpt_name.split(".")[0]
#     debug = config.get("debug", False)
#     if debug:
#         logger.setLevel(logging.DEBUG)
#         from utils.visualization.ros1_logger import LoggerROS1

#         ros1_logger = LoggerROS1("eval_debuger")

#     # TODO: remove this
#     ckpt_path = get_ckpt_path(ckpt_dir, ckpt_name, stats_path)
#     policy_config["ckpt_path"] = ckpt_path

#     # make and configure policies
#     policies: Dict[str, list] = {}
#     if ensemble is None:
#         logger.info("policy_config:", policy_config)
#         # if ensemble is not None:
#         policy_config["max_timesteps"] = max_timesteps  # TODO: remove this
#         policy = make_policy(policy_config, "eval")
#         policies["Group1"] = (policy,)
#     else:
#         logger.info("ensemble config:", ensemble)
#         ensembler = ensemble.pop("ensembler")
#         for gr_name, gr_cfgs in ensemble.items():
#             policies[gr_name] = []
#             for index, gr_cfg in enumerate(gr_cfgs):

#                 policies[gr_name].append(
#                     make_policy(
#                         gr_cfg["policies"][index]["policy_class"],
#                     )
#                 )

#     # add action filter
#     # TODO: move to policy maker as wrappers
#     if filter_type is not None:
#         # init filter
#         from OneEuroFilter import OneEuroFilter

#         config = {
#             "freq": config["fps"],  # Hz
#             "mincutoff": 0.01,  # Hz
#             "beta": 0.05,
#             "dcutoff": 0.5,  # Hz
#         }
#         filters = [OneEuroFilter(**config) for _ in range(action_dim)]

#     # init pre/post process functions
#     # TODO: move to policy maker as wrappers
#     use_stats = True
#     if use_stats:
#         with open(stats_path, "rb") as f:
#             stats = pickle.load(f)
#         pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
#         post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
#     else:
#         pre_process = lambda s_qpos: s_qpos
#         post_process = lambda a: a

#     showing_images = config.get("show_images", False)

#     def show_images(ts):
#         images: dict = ts.observation["images"]
#         for name, value in images.items():
#             # logger.info(f"Showing {name}: {value}...")
#             cv2.imshow(name, value)
#             # cv2.imwrite(f"{name}.png", value)
#         cv2.waitKey(1)

#     # evaluation loop
#     if hasattr(policy, "eval"):
#         policy.eval()
#     env_max_reward = 0
#     episode_returns = []
#     highest_rewards = []
#     num_rollouts = 0
#     policy_sig = inspect.signature(policy).parameters
#     prediction_freq = 100000
#     for rollout_id in range(max_rollouts):

#         # evaluation loop
#         all_time_actions = torch.zeros(
#             [max_timesteps, max_timesteps + num_queries, action_dim]
#         ).cuda()

#         qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
#         image_list = []  # for visualization
#         qpos_list = []
#         action_list = []
#         rewards = []
#         with torch.inference_mode():
#             logger.info("Reset environment...")
#             ts = env.reset(sleep_time=1)
#             if showing_images:
#                 # must show enough times to clear the black screen
#                 for _ in range(10):
#                     show_images(ts)
#             logger.info(f"Current rollout: {rollout_id} for {ckpt_name}.")
#             v = input(f"Press Enter to start evaluation or z and Enter to exit...")
#             if v == "z":
#                 break
#             ts = env.reset()
#             if hasattr(policy, "reset"):
#                 policy.reset()
#             try:
#                 for t in tqdm(range(max_timesteps)):
#                     start_time = time.time()
#                     image_list.append(ts.observation["images"])
#                     if showing_images:
#                         show_images(ts)
#                     # pre-process current observations
#                     curr_image = get_image(ts, camera_names, image_mode)
#                     qpos_numpy = np.array(ts.observation["qpos"])

#                     logger.debug(f"raw qpos: {qpos_numpy}")
#                     qpos = pre_process(qpos_numpy)  # normalize qpos
#                     logger.debug(f"pre qpos: {qpos}")
#                     qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
#                     qpos_history[:, t] = qpos

#                     logger.debug(f"observe time: {time.time() - start_time}")
#                     start_time = time.time()
#                     # wrap policy
#                     target_t = t % num_queries
#                     if temporal_agg or target_t == 0:
#                         # (1, chunk_size, 7) for act
#                         all_actions: torch.Tensor = policy(qpos, curr_image)
#                     all_time_actions[[t], t : t + num_queries] = all_actions
#                     index = 0 if temporal_agg else target_t
#                     raw_action = all_actions[:, index]

#                     # post-process predicted action
#                     # dim: (1,7) -> (7,)
#                     raw_action = raw_action.squeeze(0).cpu().numpy()
#                     logger.debug(f"raw action: {raw_action}")
#                     action = post_process(raw_action)  # de-normalize action
#                     # logger.debug(f"post action: {action}")
#                     if filter_type is not None:  # filt action
#                         for i, filter in enumerate(filters):
#                             action[i] = filter(action[i], time.time())
#                     # limit the prediction frequency
#                     time.sleep(max(0, 1 / prediction_freq - (time.time() - start_time)))
#                     logger.debug(f"prediction time: {time.time() - start_time}")
#                     # step the environment
#                     if debug:
#                         # dt = 1
#                         ros1_logger.log_1D("joint_position", list(qpos_numpy))
#                         ros1_logger.log_1D("joint_action", list(action))
#                         for name, image in ts.observation["images"].items():
#                             ros1_logger.log_2D("image_" + name, image)
#                     ts: dm_env.TimeStep = env.step(action, sleep_time=dt)

#                     # for visualization
#                     qpos_list.append(qpos_numpy)
#                     action_list.append(action)
#                     rewards.append(ts.reward)
#                     # debug
#                     # input(f"Press Enter to continue...")
#                     # break
#             except KeyboardInterrupt:
#                 logger.info(f"Current roll out: {rollout_id} interrupted by CTRL+C...")
#                 continue
#             else:
#                 num_rollouts += 1

#         rewards = np.array(rewards)
#         episode_return = np.sum(rewards[rewards != None])
#         episode_returns.append(episode_return)
#         episode_highest_reward = np.max(rewards)
#         highest_rewards.append(episode_highest_reward)
#         logger.info(
#             f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
#         )

#         # saving evaluation results
#         if save_dir != "":
#             dataset_name = f"{result_prefix}_{rollout_id}"
            # save_eval_results(
#                 save_dir,
#                 dataset_name,
#                 rollout_id,
#                 image_list,
#                 qpos_list,
#                 action_list,
#                 camera_names,
#                 dt,
#                 all_time_actions,
#                 save_all=save_all,
#                 save_time_actions=save_time_actions,
#             )

#     if num_rollouts > 0:
#         success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
#         avg_return = np.mean(episode_returns)
#         summary_str = (
#             f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
#         )
#         for r in range(env_max_reward + 1):
#             more_or_equal_r = (np.array(highest_rewards) >= r).sum()
#             more_or_equal_r_rate = more_or_equal_r / num_rollouts
#             summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

#         logger.info(summary_str)

#         # save success rate to txt
#         if save_dir != "":
#             with open(os.path.join(save_dir, dataset_name + ".txt"), "w") as f:
#                 f.write(summary_str)
#                 f.write(repr(episode_returns))
#                 f.write("\n\n")
#                 f.write(repr(highest_rewards))
#             logger.info(
#                 f'Success rate and average return saved to {os.path.join(save_dir, dataset_name + ".txt")}'
#             )
#     else:
#         success_rate = 0
#         avg_return = 0
#     if showing_images:
#         cv2.destroyAllWindows()
#     return success_rate, avg_return


# def eval_parser(parser: argparse.ArgumentParser = None):
#     if parser is None:
#         parser = basic_parser()
#     # change roll out num
#     parser.add_argument(
#         "-nr",
#         "--num_rollouts",
#         action="store",
#         type=int,
#         help="Maximum number of evaluation rollouts",
#         required=False,
#     )
#     # change max time steps
#     parser.add_argument(
#         "-mts",
#         "--max_timesteps",
#         action="store",
#         type=int,
#         help="max_timesteps",
#         required=False,
#     )
#     # robot config #TODO: move to robot config
#     parser.add_argument(
#         "-can",
#         "--can_buses",
#         action="store",
#         nargs="+",
#         type=str,
#         help="can_bus",
#         default=("can0", "can1"),
#         required=False,
#     )
#     parser.add_argument(
#         "-rn",
#         "--robot_name",
#         action="store",
#         type=str,
#         help="robot_name",
#         required=False,
#     )
#     parser.add_argument(
#         "-em",
#         "--eef_mode",
#         action="store",
#         nargs="+",
#         type=str,
#         help="eef_mode",
#         default=("gripper", "gripper"),
#     )
#     parser.add_argument(
#         "-bat",
#         "--bigarm_type",
#         action="store",
#         nargs="+",
#         type=str,
#         help="bigarm_type",
#         default=("OD", "OD"),
#     )
#     parser.add_argument(
#         "-fat",
#         "--forearm_type",
#         action="store",
#         nargs="+",
#         type=str,
#         help="forearm_type",
#         default=("DM", "DM"),
#     )
#     parser.add_argument(
#         "-ci",
#         "--camera_indices",
#         action="store",
#         nargs="+",
#         type=str,
#         help="camera_indices",
#         default=("0",),
#     )
#     # habitat TODO: remove this
#     parser.add_argument(
#         "-res",
#         "--habitat",
#         action="store",
#         type=str,
#         help="habitat",
#         required=False,
#     )
#     # check_images
#     parser.add_argument("-cki", "--check_images", action="store_true")
#     # set time_stamp
#     parser.add_argument(
#         "-ts",
#         "--time_stamp",
#         action="store",
#         type=str,
#         help="time_stamp",
#         required=False,
#     )
#     # save
#     parser.add_argument(
#         "-sd",
#         "--save_dir",
#         action="store",
#         type=str,
#         help="save_dir",
#         required=False,
#     )
#     parser.add_argument("-sa", "--save_all", action="store_true", help="save_all")
#     parser.add_argument(
#         "-sta", "--save_time_actions", action="store_true", help="save_time_actions"
#     )
#     # action filter type TODO: move to post process; and will use obs filter?
#     parser.add_argument(
#         "-ft",
#         "--filter",
#         action="store",
#         type=str,
#         help="filter_type",
#         required=False,
#     )
#     # yaml config path
#     parser.add_argument(
#         "-cf",
#         "--env_config_path",
#         action="store",
#         type=str,
#         help="env_config_path",
#         required=False,
#     )
#     parser.add_argument(
#         "-show",
#         "--show_images",
#         action="store_true",
#         help="show_images",
#         required=False,
#     )
#     parser.add_argument(
#         "-dbg",
#         "--debug",
#         action="store_true",
#         help="debug",
#         required=False,
#     )
#     parser.add_argument(
#         "-lts",
#         "--load_time_stamp",
#         action="store",
#         type=str,
#         help="load_time_stamp",
#         required=False,
#     )


# if __name__ == "__main__":

#     parser = basic_parser()
#     eval_parser(parser)

#     args = parser.parse_args()
#     args_dict = vars(args)
#     # TODO: put unknown key-value pairs into args_dict
#     # unknown = vars(unknown)
#     # args.update(unknown)
#     # print(unknown)
#     main(args_dict)

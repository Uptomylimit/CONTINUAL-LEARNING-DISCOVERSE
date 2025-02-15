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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args:dict):
    # get all config
    all_config = get_all_config(args, 'train')
    all_config["stage"] = "train"
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

    # 创建保存路径
    
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_dir = os.path.dirname(stats_path)
    # print(stats_dir)
    os.makedirs(stats_dir, exist_ok=True)

    # 复制配置文件到stats_dir
    shutil.copy(all_config["config_file_sys_path"], stats_dir)

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
    stats_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                              "policies/" + all_config['load_policy'],
                              "my_ckpt",
                              all_config['task_name'],
                              all_config['load_time_stamp'],
                              all_config['task_name'],
                              all_config['load_time_stamp'],
                              f'dataset_stats.pkl')

    # stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl') if stats_path == '' else stats_path
    # print(f'Saving stats into {stats_path}...')
    # with open(stats_path, 'wb') as f:
    #     pickle.dump(stats, f)

    use_stats = True
    if use_stats:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
            print(f'Loading stats from {stats_path}...')
        # 在env环境中对qpos和action进行标准化
        env.qpos_mean = stats["qpos_mean"]
        env.qpos_std = stats["qpos_std"]
        env.action_mean = stats["action_mean"]
        env.action_std = stats["action_std"]
        print("Initial mean and std",env.qpos_mean,env.qpos_std,env.action_mean,env.action_std)
        
    # 保存关键信息（must pop functions)
    key_info_path = os.path.join(stats_dir, f'key_info.pkl')
    print(f'Saving key info into {key_info_path}...')
    all_config_cp = deepcopy(all_config)
    all_config_cp["policy_config"].pop('policy_maker')
    # all_config_cp["environments"].pop('environment_maker')
    # all_config_cp["load_data"].pop("augmentors")
    key_info = {
        "init_info": {"init_joint": all_config_cp["start_joint"]},
        "all_config": all_config_cp,
    }
    with open(key_info_path, 'wb') as f:
        pickle.dump(key_info, f)

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
    best_ckpt_info = train_bc(env,env_maker, all_config)
    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def make_optimizer(policy):
    if hasattr(policy, 'configure_optimizers'):
        optimizer = policy.configure_optimizers()
    else:
        # TODO: 默认使用Adam优化器
        print('Warning: Using default optimizer')
    return optimizer

def forward_pass(data:torch.Tensor, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # print(policy)
    policy(qpos_data, image_data, action_data, is_pad)
    # print("c")
    # print("image_data.shape:", image_data.shape)
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None

def get_epoch_base(pretrain_path, epoch_base):
    if pretrain_path == "":
        epoch_base = 0
    elif epoch_base == 'AUTO':
        if pretrain_path in ["best", "last"] or "best" in pretrain_path or "last" in pretrain_path:
            epoch_base = 0
        else:
            try:
                epoch_base = int(pretrain_path.split('_')[-3])
            except:
                try:
                    epoch_base = int(pretrain_path.split('_')[-2])
                except:
                    raise ValueError(f'Invalid pretrain_ckpt_path to auto get epoch bias: {pretrain_path}')
    return epoch_base

def train_bc(env,env_maker, config):
    # num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['policy_config']
    stats_dir = os.path.dirname(config['stats_path'])
    eval_every = 3.14 if config["eval_every"] == 0 else config["eval_every"]
    parallel = config["parallel"]
    if config["eval_every"] != 0:
        from policy_evaluate import eval_bc

    # make policy and load trained policy of act
    pretrain_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                              "policies/" + config['load_policy'],
                              "my_ckpt",
                              config['task_name'],
                              config['load_time_stamp'],
                              config['task_name'],
                              config['load_time_stamp'],
                              "policy_best.ckpt")
    
    # policy_config["ckpt_path"] = pretrain_path
    policy_config["ckpt_path"] = ""
    print(f'Loading pretrained policy from {pretrain_path}...')
    # policy = make_policy(policy_config, "train")

    # # get epoch base
    # epoch_base = get_epoch_base(pretrain_path, config["pretrain_epoch_base"])

    # # make optimizer
    # optimizer = make_optimizer(policy)

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

    train_env = make_vec_env(env_maker,config, n_envs=1)  # 訓練環境
    eval_env = make_vec_env(env_maker,config, n_envs=1)   # 驗證環境
    ppo = PPO(ACT_ActorCriticPolicy,train_env,policy_config,n_steps=20,batch_size=10)
    eval_callback = EvalCallback(eval_env, best_model_save_path=config['ckpt_dir'],
                             log_path=config['ckpt_dir'], eval_freq=40,
                             n_eval_episodes=5, deterministic=True, render=False)
    ppo.learn(total_timesteps=1000,callback=eval_callback)
    # ppo.load()


    # image_list = []
    # dt = 1 / config["fps"]
    # obs,_ = env.reset()
    # # print("reset_obs:",obs["qpos"])
    # # print("reset_obs:", obs["images"])
    # for _ in range(300):
    #     action, _states = ppo.predict(obs,deterministic=True)
    #     # print(action.reshape(25,7))
    #     obs, rewards, dones, _, info = env.step(action)
    #     image_list.append(obs["images"])
    #     # print("1")
    #     # print(action.reshape(25,7))
    #     # env.render()
    #     if dones:
    #         break
    # env.close()



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




def plot_history(train_history, validation_history, num_epochs, stats_dir, seed):
    """save training curves to stats_dir"""
    for key in train_history[0]:
        plot_path = os.path.join(stats_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {stats_dir}')

def parser_train(parser:argparse.ArgumentParser = None):
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
        "-lts",
        "--load_time_stamp",
        action="store",
        type=str,
        help="load_time_stamp",
        required=False,
    )
    # set load policy name
    parser.add_argument(
        "--load_policy",
        action="store",
        type=str,
        help="load policy name",
        required=False,
    )
    return parser


if __name__ == '__main__':
    """
    参数优先级：命令行 > config文件
    """
    parser = basic_parser()
    parser_train(parser)
    main(vars(parser.parse_args()))

'''
    用一二三阶段训练好的策略初始化，
    用一二三四阶段的数据继续训练（一二三阶段数据防止遗忘一阶段策略，四阶段数据加入新的知识）
'''

from configurations.task_configs.template import (
    get_task_name,
    replace_task_name,
    set_paths,
    is_valid_module_name,
    activator,
    TASK_CONFIG_DEFAULT,
)

#看这里！！！！这是ACT的配置文件模板


def policy_maker(config: dict, stage=None):
    from policies.act.act import ACTPolicy
    from policies.common.maker import post_init_policies

    policy = ACTPolicy(config)
    # TODO: now the ckpt_path are set automatically in train and eval
    post_init_policies([policy], stage, [config["ckpt_path"]])
    return policy


def environment_maker(config: dict):
    # from envs.make_env import make_environment

    # env_config = config["environments"]
    # # TODO: use env_config only
    # return make_environment(config)
    from envs.airbot_play_3_with_cam import AIRBOTPlayWithCameraEnv

    return AIRBOTPlayWithCameraEnv(config["env_config_path"])


@activator(True)
def augment_images(image):
    from configurations.task_configs.config_augmentation.image.basic import (
        color_transforms_1,
    )

    return color_transforms_1(image)


# auto replace the task name in the default paths accoring to the file name
TASK_NAME = get_task_name(__file__)
assert is_valid_module_name(TASK_NAME), f"Invalid task name {TASK_NAME}"

replace_task_name(TASK_NAME, stats_name="dataset_stats.pkl", time_stamp="now")
# but we also show how to set the whole paths manually
# DATA_DIR = f"./data/hdf5/{TASK_NAME}"
# CKPT_DIR = f"./my_ckpt/{TASK_NAME}/ckpt"
# STATS_PATH = f"./my_ckpt/{TASK_NAME}/dataset_stats.pkl"
# set_paths(DATA_DIR, CKPT_DIR, STATS_PATH)  # replace the default data and ckpt paths

chunk_size = 25
joint_num = 7
observation_chunk_size = 1
TASK_CONFIG_DEFAULT["common"]["camera_names"] = ["0", "1"]
TASK_CONFIG_DEFAULT["common"]["state_dim"] = joint_num
TASK_CONFIG_DEFAULT["common"]["action_dim"] = joint_num
TASK_CONFIG_DEFAULT["common"]["policy_config"]["temporal_agg"] = True
TASK_CONFIG_DEFAULT["common"]["policy_config"]["chunk_size"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["observation_chunk_size"] = observation_chunk_size  #可注释掉，即只使用一个观测
TASK_CONFIG_DEFAULT["common"]["policy_config"]["num_queries"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["kl_weight"] = 10
TASK_CONFIG_DEFAULT["common"]["policy_config"]["policy_maker"] = policy_maker


TASK_CONFIG_DEFAULT["train"]["load_data"]["num_episodes"] = "ALL"
TASK_CONFIG_DEFAULT["train"]["load_data"]["batch_size_train"] = 512
TASK_CONFIG_DEFAULT["train"]["load_data"]["batch_size_validate"] = 512
TASK_CONFIG_DEFAULT["train"]["load_data"]["observation_slice"] = None
TASK_CONFIG_DEFAULT["train"]["load_data"]["action_slice"] = None
TASK_CONFIG_DEFAULT["train"]["num_epochs"] = 2000
TASK_CONFIG_DEFAULT["train"]["validate_every"] = 100
TASK_CONFIG_DEFAULT["train"]["learning_rate"] = 1e-5
TASK_CONFIG_DEFAULT["train"]["pretrain_ckpt_path"] = ""
TASK_CONFIG_DEFAULT["train"]["pretrain_epoch_base"] = "AUTO"

TASK_CONFIG_DEFAULT["eval"]["robot_num"] = 1
TASK_CONFIG_DEFAULT["eval"]["joint_num"] = joint_num
TASK_CONFIG_DEFAULT["eval"]["start_joint"] = "AUTO"
TASK_CONFIG_DEFAULT["eval"]["max_timesteps"] = 700
TASK_CONFIG_DEFAULT["eval"]["ensemble"] = None
TASK_CONFIG_DEFAULT["eval"]["environments"]["environment_maker"] = environment_maker
TASK_CONFIG_DEFAULT["eval"]["ckpt_names"] = ["policy_best.ckpt"]


# ##################cosine_annealing
# TASK_CONFIG_DEFAULT["train"]["use_cosine_annealing"] = True
# TASK_CONFIG_DEFAULT["train"]['lr_min'] = 1e-7
# #########################


# final config
TASK_CONFIG = TASK_CONFIG_DEFAULT

# from configurations.task_configs.template import (
#     get_task_name,
#     replace_task_name,
#     set_paths,
#     is_valid_module_name,
#     activator,
#     TASK_CONFIG_DEFAULT,
# )


# def policy_maker(config: dict, stage=None):
#     from policies.diffusion.diffusion_policy import DiffusionPolicy
#     from policies.common.maker import post_init_policies

#     policy = DiffusionPolicy(config)
#     # TODO: now the ckpt_path are set automatically in train and eval
#     post_init_policies([policy], stage, [config["ckpt_path"]])
#     return policy


# def environment_maker(config: dict):
#     from envs.make_env import make_environment

#     env_config = config["environments"]
#     # TODO: use env_config only
#     return make_environment(config)


# @activator(True)
# def augment_images(image):
#     from configurations.task_configs.config_augmentation.image.basic import (
#         color_transforms_1,
#     )

#     return color_transforms_1(image)


# # auto replace the task name in the default paths accoring to the file name
# TASK_NAME = get_task_name(__file__)
# assert is_valid_module_name(TASK_NAME), f"Invalid task name {TASK_NAME}"

# replace_task_name(TASK_NAME, stats_name="dataset_stats.pkl", time_stamp="now")
# # but we also show how to set the whole paths manually
# # DATA_DIR = f"./data/hdf5/{TASK_NAME}"
# # CKPT_DIR = f"./my_ckpt/{TASK_NAME}/ckpt"
# # STATS_PATH = f"./my_ckpt/{TASK_NAME}/dataset_stats.pkl"
# # set_paths(DATA_DIR, CKPT_DIR, STATS_PATH)  # replace the default data and ckpt paths

# prediction_horizon = 16
# joint_num = 7
# TASK_CONFIG_DEFAULT["common"]["camera_names"] = ["0", "1"]
# TASK_CONFIG_DEFAULT["common"]["state_dim"] = joint_num
# TASK_CONFIG_DEFAULT["common"]["action_dim"] = joint_num

# #TODO: set the diffusion policy config
# POLICY_CONFIG_DIFFUSION_DEFAULT = {
#     # transformer args
#     # "hidden_dim": 512,
#     # "dim_feedforward": 3200,
#     # "enc_layers": 4,
#     # "dec_layers": 7,
#     # "nheads": 8,
#     'observation_horizon': 2,
#     'action_horizon': 8,
#     'prediction_horizon': 16,
#     'num_queries': 8,
#     # scheduler args
#     'num_train_timesteps': 1000,
#     'num_inference_timesteps': 10,
#     'beta_schedule': 'squaredcos_cap_v2',
#     'prediction_type': 'epsilon',
#     'clip_sample': True,
#     'ema_power': 0.75,
#     "policy_class": "Diffusion", 
#     "policy_maker": policy_maker,
# }
# TASK_CONFIG_DEFAULT["common"]["policy_config"] = POLICY_CONFIG_DIFFUSION_DEFAULT


# TASK_CONFIG_DEFAULT["train"]["load_data"]["num_episodes"] = "ALL"
# TASK_CONFIG_DEFAULT["train"]["load_data"]["batch_size_train"] = 4
# TASK_CONFIG_DEFAULT["train"]["load_data"]["batch_size_validate"] = 4
# TASK_CONFIG_DEFAULT["train"]["load_data"]["observation_slice"] = None
# TASK_CONFIG_DEFAULT["train"]["load_data"]["action_slice"] = None
# TASK_CONFIG_DEFAULT["train"]["num_epochs"] = 100
# TASK_CONFIG_DEFAULT["train"]["validate_every"] = 50
# TASK_CONFIG_DEFAULT["train"]["learning_rate"] = 2e-5
# TASK_CONFIG_DEFAULT["train"]["pretrain_ckpt_path"] = ""
# TASK_CONFIG_DEFAULT["train"]["pretrain_epoch_base"] = "AUTO"

# TASK_CONFIG_DEFAULT["eval"]["robot_num"] = 1
# TASK_CONFIG_DEFAULT["eval"]["joint_num"] = joint_num
# TASK_CONFIG_DEFAULT["eval"]["start_joint"] = "AUTO"
# TASK_CONFIG_DEFAULT["eval"]["max_timesteps"] = 300
# TASK_CONFIG_DEFAULT["eval"]["ensemble"] = None
# TASK_CONFIG_DEFAULT["eval"]["environments"]["environment_maker"] = environment_maker
# TASK_CONFIG_DEFAULT["eval"]["ckpt_names"] = ["policy_best.ckpt"]

# # final config
# TASK_CONFIG = TASK_CONFIG_DEFAULT

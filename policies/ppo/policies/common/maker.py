import logging, os
import torch
from typing import List


def make_policy(
    config, stage=None
):  # TODO: remove this function and use the config file
    policy_maker = config["policy_maker"]
    policy = policy_maker(config, stage)
    assert policy is not None, "Please use the make_policy function in the config file"
    return policy


def post_init_policies(policies: List[torch.nn.Module], stage, ckpt_paths) -> None:
    """Load the checkpoint for the policies and move them to the GPU
    Args:
        policies (List[torch.nn.Module]): List of policies
        stage (str): "train" or "eval"
        ckpt_paths (List[str]): List of checkpoint paths
    """
    # https://pytorch.org/docs/stable/generated/torch.load.html
    weights_only = False
    for policy, ckpt_path in zip(policies, ckpt_paths):
        if ckpt_path not in [None, ""]:
            if not os.path.exists(ckpt_path):
                raise Exception(f"Checkpoint path does not exist: {ckpt_path}")
            if stage == "train":
                load = torch.load(ckpt_path, weights_only=weights_only)
                # names = []
                # for name, param in load.items():
                #     names.append(name)
                #     # print(f"Values: {param}")
                # print(names)
                fixed_state_dict = {k.replace('module.', ''): v for k, v in load.items()}

                loading_status = policy.load_state_dict(fixed_state_dict)
                logging.info(f'Resume policy from: {ckpt_path}, Status: {loading_status}')
            elif stage == "eval":
                load = torch.load(ckpt_path, weights_only=weights_only)

                # names = []
                # for name, param in load.items():
                #     names.append(name)
                #     # print(f"Values: {param}")
                # print(names)

                # 修改参数键名，移除 'module.' 前缀 由于dataparalell的封装
                fixed_state_dict = {k.replace('module.', ''): v for k, v in load.items()}
                    
                loading_status = policy.load_state_dict(fixed_state_dict)
                logging.info(loading_status)
                logging.info(f"Loaded: {ckpt_path}")
        policy.cuda()

        if stage == "eval":
            policy.eval()

def save_model(policy, path):
    torch.save(policy.state_dict(), path)
    logging.info(f"Saved policy to: {path}")
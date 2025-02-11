# 这个是第3阶段
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp

import traceback
from discoverse.airbot_play import AirbotPlayFIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
from discoverse.envs.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play

class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.camera_1_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())
        # 获取需要随机化的积木块 body 名称列表
        self.block_body_names = ["block1_green", "block2_green", "block_purple1", "block_purple2", "block_purple3", "block_purple4", "block_purple5", "block_purple6"]
        # 存储原始的动力学参数，用于设定随机化范围
        self.original_mass = {}
        self.original_friction = {}
        self.original_geom_pos = {} # 存储 geom 的原始 pos

        for body_name in self.block_body_names:
            body_id = self.mj_model.body(body_name).id
            self.original_mass[body_name] = self.mj_model.body_mass[body_id].copy()
            self.original_friction[body_name] = []
            self.original_geom_pos[body_name] = []

            for geom_id in range(self.mj_model.body_geomnum[body_id]):
                geom_friction = self.mj_model.geom_friction[self.mj_model.body_geomadr[body_id] + geom_id].copy()
                self.original_friction[body_name].append(geom_friction)

                geom_pos = self.mj_model.geom_pos[self.mj_model.body_geomadr[body_id] + geom_id].copy()
                self.original_geom_pos[body_name].append(geom_pos)
    def domain_randomization(self):

        # --- 随机化积木块动力学参数 ---
        for body_name in self.block_body_names:
            body_id = self.mj_model.body(body_name).id

            # 1. 随机化质量
            mass_range_ratio = 0.1 # 质量随机范围 ±20%
            min_mass = self.original_mass[body_name] * (1.0 - mass_range_ratio)
            max_mass = self.original_mass[body_name] * (1.0 + mass_range_ratio)
            random_mass = np.random.uniform(min_mass, max_mass)
            self.mj_model.body_mass[body_id] = random_mass





            # 2. 随机化摩擦系数 (只随机化滑动摩擦 friction_slide)
            friction_range_ratio_lower = 0.8 # 摩擦系数随机范围 ±50%
            friction_range_ratio_upper = 1.3 # 摩擦系数随机范围 ±50%
            rollfriction_range_ratio_lower = 0.75 # 摩擦系数随机范围 ±50%
            rollfriction_range_ratio_upper = 2.0 # 摩擦系数随机范围 ±50%
            for geom_idx in range(len(self.original_friction[body_name])):
                original_friction = self.original_friction[body_name][geom_idx]
                min_friction = original_friction * friction_range_ratio_lower
                max_friction = original_friction * friction_range_ratio_upper 
                random_friction = np.random.uniform(min_friction[0], max_friction[0])
                random_friction2 = np.random.uniform(min_friction[1], max_friction[1])
                min_friction = original_friction * rollfriction_range_ratio_lower
                max_friction = original_friction * rollfriction_range_ratio_upper 
                random_friction3 = np.random.uniform(min_friction[2], max_friction[2]) 

                self.mj_model.geom_friction[self.mj_model.body_geomadr[body_id] + geom_idx][0] = random_friction # 修改静摩擦
                self.mj_model.geom_friction[self.mj_model.body_geomadr[body_id] + geom_idx][1] = random_friction2 # 修改滑动摩擦
                self.mj_model.geom_friction[self.mj_model.body_geomadr[body_id] + geom_idx][2] = random_friction3 # 修改滚动摩擦


            # 3. 随机化质心 (通过轻微随机偏移 geom 位置)
            com_pos_range = 0.001 # 质心偏移范围 ±0.001 米
            for geom_idx in range(len(self.original_geom_pos[body_name])):
                original_geom_pos = self.original_geom_pos[body_name][geom_idx]
                random_pos_offset = np.random.uniform(-com_pos_range, com_pos_range, size=3)
                self.mj_model.geom_pos[self.mj_model.body_geomadr[body_id] + geom_idx] = original_geom_pos + random_pos_offset

        # 随机 2个绿色长方体位置

        for z in range(2):
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 0] += (
                10.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 1] += (
                10.0 * (np.random.random() - 0.5) * 0.001
            )

        # 随机 10个紫色方块位置

        for z in range(6):
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )

    def check_success(self):
        tmat_bridge1 = get_body_tmat(self.mj_data, "bridge1")
        tmat_bridge2 = get_body_tmat(self.mj_data, "bridge2")
        tmat_block1 = get_body_tmat(self.mj_data, "block1_green")
        tmat_block2 = get_body_tmat(self.mj_data, "block2_green")
        tmat_block01 = get_body_tmat(self.mj_data, "block_purple3")
        tmat_block02 = get_body_tmat(self.mj_data, "block_purple6")
        return ((abs(tmat_block1[2, 2]) < 0.001) 
                and (abs(abs(tmat_bridge1[1,3] - tmat_bridge2[1,3]) - 0.03) <= 0.002) 
                and (abs(tmat_block2[2, 2]) < 0.001))

cfg = AirbotPlayCfg()
cfg.use_gaussian_renderer = False
cfg.init_key = "ready"
# cfg.gs_model_dict["background"]  = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["background"]  = "/home/hdl/DISCOVERSE/models/3dgs/scene/ours_scene/scene30.ply"
cfg.gs_model_dict["drawer_1"]    = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]    = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"]   = "object/bowl_pink.ply"
cfg.gs_model_dict["bridge1"] = "object/bridge_es2.ply"
cfg.gs_model_dict["bridge2"] = "object/bridge_es2.ply"
cfg.gs_model_dict["block_purple1"] = "object/block_purple_es.ply"
cfg.gs_model_dict["block_purple2"] = "object/block_purple_es.ply"
cfg.gs_model_dict["block_purple3"] = "object/block_purple_es.ply"
cfg.gs_model_dict["block_purple4"] = "object/block_purple_es.ply"
cfg.gs_model_dict["block_purple5"] = "object/block_purple_es.ply"
cfg.gs_model_dict["block_purple6"] = "object/block_purple_es.ply"
cfg.gs_model_dict["block1_green"] = "object/green_long_es.ply"
cfg.gs_model_dict["block2_green"] = "object/green_long_es.ply"

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/block_bridge_place1.xml"
cfg.obj_list     = ["bridge1","bridge2","block1_green","block2_green", "block_purple1", "block_purple2", "block_purple3", "block_purple4", "block_purple5", "block_purple6"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = True
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 448,
    "height" : 448
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1000, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False


    save_dir = "/media/hdl/My_Passport/Tsinghua_air/data_create/state1"
#   save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/block_bridge_place_realtest")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        shutil.copyfile(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))
        
    arm_fik = AirbotPlayFIK(os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"))

    trmat = Rotation.from_euler("xyz", [0., np.pi/2, 0.], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 79
    max_time = 70.0 # seconds
    
    action = np.zeros(7)
    process_list = []

    move_speed = 0.75
    sim_node.reset()

    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []

        try:
            if stm.trigger():

                if stm.state_idx == 0: # 伸到拱桥上方
                    trmat = Rotation.from_euler("xyz", [0., np.pi/2, np.pi/2], degrees=False).as_matrix()
                    tmat_bridge1 = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge1[:3, 3] = tmat_bridge1[:3, 3] + np.array([0.03, -0.015, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge1
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 1: # 伸到长方体上方
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 3: # 伸到长方体
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array([0, 0, 0.04])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 4: # 抓住长方体
                    sim_node.target_control[6] = 0.29
                elif stm.state_idx == 5: # 抓稳长方体
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 6: # 提起长方体
                    tmat_tgt_local[2,3] += 0.09
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 7: # 把长方体放到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3,3] = tmat_bridge[:3, 3] + np.array([0.075 + 0.00005, -0.015, 0.1])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])                  
                elif stm.state_idx == 8: # 保持夹爪角度 降低高度 把长方体放到桥旁边
                    tmat_tgt_local[2,3] -= 0.03
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 9: # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 10: # 抬升高度
                    tmat_tgt_local[2,3] += 0.06
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])


                elif stm.state_idx == 11: # 伸到拱桥上方
                    tmat_bridge1 = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge1[:3, 3] = tmat_bridge1[:3, 3] + np.array([0.03, -0.015, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge1
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 1                
                elif stm.state_idx == 12: # 伸到长方体上方
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 1 
                elif stm.state_idx == 14: # 伸到长方体
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array([0, 0, 0.04])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 15: # 抓住长方体
                    sim_node.target_control[6] = 0.29
                elif stm.state_idx == 16: # 抓稳长方体
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 17: # 提起长方体
                    tmat_tgt_local[2,3] += 0.09
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 18: # 把长方体放到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3,3] = tmat_bridge[:3, 3] + np.array([-0.015 - 0.0005, -0.015, 0.1])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])                  
                elif stm.state_idx == 19: # 保持夹爪角度 降低高度 把长方体放到桥旁边
                    tmat_tgt_local[2,3] -= 0.03
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 20: # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 21: # 抬升高度
                    tmat_tgt_local[2,3] += 0.06
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])

                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            traceback.print_exc()
            sim_node.reset()

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(target=recoder_airbot_play, args=(save_path, act_lst, obs_lst, cfg))
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()

    for p in process_list:
        p.join()

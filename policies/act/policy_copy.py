import pickle
from airbot_py.airbot_play import AirbotPlay
from time import sleep
def load_action_list(file_path):
    with open(file_path, 'rb') as f:
        action_list = pickle.load(f)
    return action_list

# 示例用法
file_path = 'policies/act/eval_results/state4_random1000/20250221-172332/actions_0.pkl'
action_list = load_action_list(file_path)

robot = AirbotPlay(ip="localhost",port=50051)

robot.set_target_joint_q([0,0,0,0,0,0],blocking=True)
robot.set_target_end(0, blocking=True)
for action in action_list:
    robot.set_target_joint_q(action[:6],blocking=True)
    robot.set_target_end(action[6], blocking=True)
    print("log:", action,robot.get_current_joint_q(),robot.get_current_end())
    sleep(0.1)
robot.set_target_joint_q([0,0,0,0,0,0],blocking=True)
robot.set_target_end(0, blocking=True)

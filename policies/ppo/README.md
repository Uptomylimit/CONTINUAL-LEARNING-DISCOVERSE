下载stablebaselines3

pip install stablebaselines3

训练指令

python3 policies/train.py ppo --policy ppo -tn block_bridge_place_1 --load_policy act -lts 20250211-164319 -rn discoverse/examples/tasks_airbot_play/block_bridge_place_1

其中，--policy指定策略，-tn指定任务，--load_policy指定导入的模型，-lts指定导入模型的timestamp，-rn指定simnode的位置

评估指令

python3 policies/infer.py ppo --policy ppo -tn block_bridge_place_1 -lts 20250214-191932 -rn discoverse/examples/tasks_airbot_play/block_bridge_place_1

其中，--policy指定策略，-tn指定任务，没有--load_policy，默认导入ppo，-lts指定导入模型的timestamp，-rn指定simnode的位置
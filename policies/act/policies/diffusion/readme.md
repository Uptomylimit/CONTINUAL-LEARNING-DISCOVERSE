### Install robomimic
everything is in the robomimic file folder, just *cd robomimic* and run *pip install -e .* in robomimic

### Install others needed package by yourself

### usage: 和原来基本保持一样

#### model：
To use RDT, you should set *model_type* in the task config file(block_bridge_place_diffusion.py), or you can use set 'Unet' to the *model_type*.

#### train:
python3 policies/train.py diffusion -tn block_bridge_place_diffusion
#### inference:
python3 policies/infer.py diffusion -tn block_bridge_place_diffusion  -mts 100 -ts 20250213-112342 -rn discoverse/examples/tasks_airbot_play/block_bridge_place.py 
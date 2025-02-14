import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import re

from ..common.models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from ..common.models.rdt.rdt_model import RDT
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.camera_names = args_override['camera_names']

        self.observation_horizon = args_override['observation_horizon']
        self.action_horizon = args_override['action_horizon'] # apply chunk size
        self.prediction_horizon = args_override['prediction_horizon'] # chunk size
        self.num_train_timesteps = args_override['num_train_timesteps']
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.prediction_type = args_override['prediction_type']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.model_type = args_override["model_type"]
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim'] # 14 + 2
        self.obs_dim = self.feature_dimension * len(self.camera_names) # camera features
        
        self.image_num = len(self.camera_names) * self.observation_horizon
    
        if self.model_type == 'unet':
            backbones = []
            pools = []
            linears = []
            for _ in range(self.image_num):
                backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': True, 'input_coord_conv': False}))
                pools.append(SpatialSoftmax(**{'input_shape': [512, 14, 14], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
                linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
            backbones = nn.ModuleList(backbones)
            pools = nn.ModuleList(pools)
            linears = nn.ModuleList(linears)
            
            backbones = replace_bn_with_gn(backbones) # TODO


            noise_pred_net = ConditionalUnet1D(
                input_dim=self.ac_dim,
                global_cond_dim=self.obs_dim*self.observation_horizon + args_override['state_dim']
            )

            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'backbones': backbones,
                    'pools': pools,
                    'linears': linears,
                    'noise_pred_net': noise_pred_net
                })
            })
            nets = nets.float().cuda()


        elif self.model_type == 'RDT':
            self.model_args = args_override["model_args"]
            self.vision_encoder = SiglipVisionTower(vision_tower=self.model_args['pretrained_vision_encoder_name_or_path'], args=None)
            self.image_processor = self.vision_encoder.image_processor
            img_adaptor = self.build_condition_adapter(self.model_args['img_adaptor'], in_features=self.model_args['img_token_dim'], out_features=self.model_args['hidden_size'])
            state_adaptor = self.build_condition_adapter(self.model_args['img_adaptor'], in_features=self.model_args['state_token_dim'], out_features=self.model_args['hidden_size'])
            model = RDT(
                output_dim=self.ac_dim,
                horizon=self.prediction_horizon,
                hidden_size=self.model_args['hidden_size'],
                depth=self.model_args['depth'],
                num_heads=self.model_args['num_heads'],
                #    max_lang_cond_len=max_lang_cond_len,
                img_cond_len=(args_override["observation_horizon"]
                        * len(self.camera_names) * self.vision_encoder.num_patches), 
                #    lang_pos_embed_config=lang_pos_embed_config,
                img_pos_embed_config=[
                        ("image", (args_override["observation_horizon"], 
                                len(self.camera_names), 
                                -self.vision_encoder.num_patches)),  
                        ],
                dtype=torch.bfloat16,
                )
            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'img_adaptor': img_adaptor,
                    'state_adaptor': state_adaptor,
                    'noise_pred_net': model,
                })
            })
            nets = nets.to(device='cuda', dtype=torch.bfloat16)

        else:
            print(f"model type {self.model_type} not implemented")
            raise NotImplementedError 

        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(parameters=nets.parameters(), power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args_override['num_train_timesteps'],
            beta_schedule=args_override['beta_schedule'],
            clip_sample=args_override['clip_sample'],
            prediction_type=args_override['prediction_type'],
        )

        self.noise_scheduler_sample = DDIMScheduler(
            num_train_timesteps=args_override['num_train_timesteps'],
            beta_schedule=args_override['beta_schedule'],
            clip_sample=args_override['clip_sample'],
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=args_override['prediction_type'],
        )
        # other solvers
        # self.noise_scheduler = DPMSolverMultistepScheduler(
        #     num_train_timesteps=args_override['num_inference_timesteps'],
        #     beta_schedule=args_override['squaredcos_cap_v2'],
        #     clip_sample=args_override['clip_sample'],
        #     prediction_type=args_override['epsilon'],
        # )
        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_img = self.nets['policy']['img_adaptor'](img_tokens)
        adpated_state = self.nets['policy']['state_adaptor'](state_tokens)

        return adpated_img, adpated_state

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None: # training time
            if self.model_type == 'RDT':
                qpos = qpos.unsqueeze(1)
                batch_size, _, C, H, W = image.shape
                image = self.image_processor.preprocess(image.reshape(-1, C, H, W), return_tensors='pt')['pixel_values']
                image_embeds = self.vision_encoder(image).detach()
                image_embeds = image_embeds.reshape((batch_size, -1, self.vision_encoder.hidden_size))
                # sample noise to add to actions
                noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device)
                
                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, self.num_train_timesteps, 
                    (B,), device=actions.device
                ).long()
                
                # add noise to the clean actions according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self.noise_scheduler.add_noise(
                    actions, noise, timesteps)

                qpos_actions_traj = torch.cat([qpos, noisy_actions], dim=1)

                actions = actions.to(dtype=torch.bfloat16)
                noise = noise.to(dtype=torch.bfloat16)
                image_embeds = image_embeds.to(dtype=torch.bfloat16)
                qpos_actions_traj = qpos_actions_traj.to(dtype=torch.bfloat16)

                img_cond, qpos_actions_traj = self.adapt_conditions(image_embeds, qpos_actions_traj)


                noise_pred = self.nets['policy']['noise_pred_net'](qpos_actions_traj, 
                                                              timesteps, img_cond)

            elif self.model_type == 'Unet':

                #encode conditions
                all_features = []
                for cam_id in range(self.image_num):
                    cam_image = image[:, cam_id]
                    cam_features = self.nets['policy']['backbones'][cam_id](cam_image)
                    pool_features = self.nets['policy']['pools'][cam_id](cam_features)
                    pool_features = torch.flatten(pool_features, start_dim=1)
                    out_features = self.nets['policy']['linears'][cam_id](pool_features)
                    all_features.append(out_features)

                obs_cond = torch.cat(all_features + [qpos], dim=1)

                # sample noise to add to actions
                noise = torch.randn(actions.shape, dtype=actions.dtype, device=obs_cond.device)
                
                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, self.num_train_timesteps, 
                    (B,), device=obs_cond.device
                ).long()
                
                # add noise to the clean actions according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self.noise_scheduler.add_noise(
                    actions, noise, timesteps)
            
                # predict the noise residual
                noise_pred = self.nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            #other models
            else:
                print(f"model type {self.model_type} not implemented")
                raise NotImplementedError

            # L2 loss
            if self.prediction_type == 'epsilon':
                target = noise
            elif self.prediction_type == 'sample':
                target = actions
            all_l2 = F.mse_loss(noise_pred, target, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(self.nets.parameters())
            return loss_dict
        else: # inference time
            To = self.observation_horizon
            # Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            if self.ema is not None:
                self.ema.copy_to(self.nets.parameters())
            nets = self.nets
            naction = None
            if self.model_type == 'RDT':
                qpos = qpos.unsqueeze(1)
                batch_size, _, C, H, W = image.shape
                image = self.image_processor.preprocess(image.reshape(-1, C, H, W), return_tensors='pt')['pixel_values']
                image_embeds = self.vision_encoder(image).detach()
                image_embeds = image_embeds.reshape((batch_size, -1, self.vision_encoder.hidden_size))

                # initialize action from Guassian noise
                noisy_actions = torch.randn(
                    (B, Tp, action_dim), device=image_embeds.device, dtype=torch.bfloat16)
                naction = noisy_actions
                
                image_embeds = image_embeds.to(dtype=torch.bfloat16)
                qpos = qpos.to(dtype=torch.bfloat16)

                img_cond, qpos_traj = self.adapt_conditions(image_embeds, qpos)

                self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

                for k in self.noise_scheduler_sample.timesteps:
                    action_traj = nets['policy']['state_adaptor'](naction)
                    qpos_actions_traj = torch.cat([qpos_traj, action_traj], dim=1)

                    noise_pred = nets['policy']['noise_pred_net'](qpos_actions_traj,
                                                     k.unsqueeze(-1).to(device=image_embeds.device),
                                                     img_cond)
                    
                    naction = self.noise_scheduler_sample.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                    naction = naction.to(dtype=qpos_traj.dtype)

            elif self.model_type == 'Unet':
                all_features = []
                for cam_id in range(len(self.camera_names)*To):
                    cam_image = image[:, cam_id]
                    cam_features = nets['policy']['backbones'][cam_id](cam_image)
                    pool_features = nets['policy']['pools'][cam_id](cam_features)
                    pool_features = torch.flatten(pool_features, start_dim=1)
                    out_features = nets['policy']['linears'][cam_id](pool_features)
                    all_features.append(out_features)

                obs_cond = torch.cat(all_features + [qpos], dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, Tp, action_dim), device=obs_cond.device)
                naction = noisy_action
            
                # init scheduler
                self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

                for k in self.noise_scheduler_sample.timesteps:
                    # predict noise
                    noise_pred = nets['policy']['noise_pred_net'](
                        sample=naction, 
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler_sample.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            return naction

    def state_dict(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def load_state_dict(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status

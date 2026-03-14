import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from training.adastep import HorizonPredictor, AdaptiveHorizonLoss
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')
        
        # AdaStep 相关配置
        self.use_adastep = args_override.get('use_adastep', False)
        if self.use_adastep:
            self.horizon_predictor = HorizonPredictor(
                input_dim=args_override.get('hidden_dim', 512),
                hidden_dim=256
            )
            self.k_min = args_override.get('k_min', 5)
            self.k_max = args_override.get('k_max', 50)
            self.horizon_weight = args_override.get('horizon_weight', 1.0)
            
            # 联合损失函数
            self.adaptive_loss = AdaptiveHorizonLoss(
                kl_weight=self.kl_weight,
                horizon_weight=self.horizon_weight
            )
            print(f'✓ AdaStep 已启用: k_min={self.k_min}, k_max={self.k_max}')
        else:
            self.horizon_predictor = None
            print('✗ AdaStep 未启用（使用固定步长）')

    def __call__(self, qpos, image, actions=None, is_pad=None, horizon_labels=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
            loss_dict = dict()
            
            if self.use_adastep and horizon_labels is not None:
                # 提取 latent feature（假设是 mu）
                latent_feature = mu
                
                # 预测步长
                horizon_pred = self.horizon_predictor(latent_feature)
                
                # 使用联合损失
                horizon_labels_tensor = horizon_labels.to(qpos.device)
                loss_dict = self.adaptive_loss.forward(
                    action_pred=a_hat,
                    action_gt=actions,
                    is_pad=is_pad,
                    kl_loss=total_kld[0],
                    horizon_pred=horizon_pred,
                    horizon_gt=horizon_labels_tensor
                )
            else:
                # 原始损失
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict['l1'] = l1
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            
            return loss_dict
        else: # inference time
            a_hat, _, (mu, _) = self.model(qpos, image, env_state) # no action, sample from prior
            
            # 如果启用 AdaStep，同时返回预测的步长
            if self.use_adastep:
                latent_feature = mu
                predicted_horizon = self.horizon_predictor.predict_horizon(
                    latent_feature, self.k_min, self.k_max
                )
                return a_hat, predicted_horizon
            else:
                return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        """Return a serializable state for saving.

        Includes the backbone ACT model and any attached AdaStep predictor (if enabled).
        Keeping this as `state_dict()` ensures compatibility with existing save/load
        code that expects a single dict from `policy.serialize()`.
        """
        return self.state_dict()

    def deserialize(self, model_dict):
        """Load a state previously returned by `serialize()`.

        Returns the result of `load_state_dict` so callers can inspect missing/extra keys.
        """
        return self.load_state_dict(model_dict)


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
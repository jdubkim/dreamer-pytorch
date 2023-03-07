import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np


class DiscreteActionModel(nn.Module):
    def __init__(
        self,
        action_size: int,
        deter_size: int,
        stoch_size: int,
        embedding_size: int,
        actor_info: dict,
        expl_info: dict,
    ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
        self.n_layers = actor_info["n_layers"]
        self.node_size = actor_info["node_size"]
        self.activation_fn = actor_info["activation"]
        self.dist = actor_info["dist"]
        self.train_noise = expl_info["train_noise"]
        self.eval_noise = expl_info["eval_noise"]
        self.expl_min = expl_info["expl_min"]
        self.expl_decay = expl_info["expl_decay"]
        self.expl_type = expl_info["expl_type"]
        self.model = self._build_model()

    def _build_model(self):
        model = [
            nn.Linear(
                self.deter_size + self.stoch_size, self.node_size
            ),
            self.activation_fn(),
        ]
        for _ in range(self.layers - 1):
            model += [
                nn.Linear(self.node_size, self.node_size),
                self.activation_fn(),
            ]
        
        if self.dist == 'one_hot':
            model += [nn.Linear(self.node_size, self.action_size)]
        else:
            raise NotImplementedError
        
        return nn.Sequential(*model)

    def foward(self, model_state):
        action_dist = self.get_action_dist(model_state)
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach()
        return action, action_dist

    def get_action_dist(self, model_state):
        logits = self.model(model_state)
        if self.dist == 'one_hot':
            return td.OneHotCategorical(logits=logits)
        else:
            raise NotImplementedError

    def add_exploration(self, action: torch.Tensor, iter: int, mode: str = 'train'):
        if mode == 'train':
            expl_noise = self.train_noise
            expl_noise = expl_noise - iter / self.expl_decay
            expl_noise = max(expl_noise, self.expl_min)
        elif mode == 'eval':
            expl_noise = self.eval_noise
        else:
            raise NotImplementedError
        
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_noise:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[:, index] = 1
            return action

        raise NotImplementedError

                
            
            
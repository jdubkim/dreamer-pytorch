import torch
import torch.nn as nn
from dreamerv2.utils.rssm import RSSMUtils, RSSMContState, RSSMDiscState


class RSSM(nn.Module, RSSMUtils):
    def __init__(
        self,
        action_size: int,
        rssm_node_size: int,
        embedding_size: int,
        device: torch.device,
        rssm_type: str,
        info: dict,
        activation_fn: nn.Module = nn.ELU,
    ) -> None:
        nn.Module.__init__(self)
        RSMUtils.__init__(self, rssm_type=rssm_type, info=info)
        self.device = device
        self.action_size = action_size
        self.rssm_node_size = rssm_node_size
        self.embedding_size = embedding_size
        self.activation_fn = activation_fn
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.fc_embed_state_action = self._build_embed_state_action()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()

    def _build_embed_state_action(self) -> nn.Module:
        fc_embed_state_action = [
            nn.Linear(self.stoch_size + self.action_size, self.deter_size),
            self.activation_fn(),
        ]
        return nn.Sequential(*fc_embed_state_action)

    def _build_temporal_prior(self) -> nn.Module:
        """
        Given the latest deterministic state, output prior over stochastic state.
        """
        temporal_prior = [
            nn.Linear(self.deter_size, self.rssm_node_size),
            self.activation_fn(),
        ]
        if self.rssm_type == "discrete":
            temporal_prior += [nn.Linear(self.rssm_node_size, self.stoch_size)]
        elif self.rssm_type == "continuous":
            temporal_prior += [nn.Linear(self.rssm_node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_prior)

    def _build_temporal_posterior(self) -> nn.Module:
        """
        Given the latest embedded observation and determinsitic state,
        output posteior over stochastic states.
        """
        temporal_posterior = [
            nn.Linear(self.deter_size + self.embedding_size, self.rssm_node_size),
            self.activation_fn(),
        ]

        if self.rssm_type == "discrete":
            temporal_posterior += [nn.Linear(self.rssm_node_size, self.stoch_size)]
        elif self.rssm_type == "continuous":
            temporal_posterior += [nn.Linear(self.rssm_node_size, 2 * self.stoch_size)]

        return nn.Sequential(*temporal_posterior)

    def imagine(
        self,
        prev_action: torch.Tensor,
        prev_rssm_state: RSSMState,
        nonterms: bool = True,
    ) -> RSSMState:
        state_action_embed = self.fc_embed_state_action(
            torch.cat([prev_rssm_state.stoch * nonterms, prev_action], dim=-1)
        )
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter * nonterms)

        if self.rssm_type == "discrete":
            prior_logit = self.fc_prior(deter_state)
            stats = {"logit": prior_logit}
            prior_stoch_state = self.get_sotch_state(stats)
            prior_rssm_state = RSSMDiscState(
                prior_logit, prior_stoch_state, deter_state
            )

        elif self.rssm_type == "continuous":
            prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
            stats = {"mean": prior_mean, "std": prior_std}
            prior_stoch_state, std = self.get_sotch_state(stats)
            prior_rssm_state = RSSMContState(
                prior_mean, std, prior_stoch_state, deter_state
            )

        return prior_rssm_state

    def rollout_imagination(
        self, horizon: int, actor: nn.Module, prev_rssm_state: RSSMState
    ) -> tuple:
        rssm_state = prev_rssm_state
        next_rssm_states = []
        imag_log_probs = []
        action_entropy = []

        for _ in range(horizon):
            action, action_dist = actor((self.get_model_state(rssm_state)).detach())
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))
            action_entropy.append(action_dist.entropy())
        
        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)

        return next_rssm_states, imag_log_probs, action_entropy

    def rssm_observe(self, obs_embed: torch.Tensor, prev_action: torch.Tensor, prev_nonterm: bool, prev_rssm_state: RSSMState) -> :
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        if self.rssm_type == "discrete":
            posterior_logit = self.fc_posterior(x)
            stats = {"logit": posterior_logit}
            posterior_stoch_state = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state)
        
        elif self.rssm_type == "continuous":
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
            stats = {"mean": posterior_mean, "std": posterior_std}
            posterior_stoch_state, std = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state)
        
        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(self, horizon: int, obs_embedding: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state: RSSMState) -> tuple:
        priors = []
        posteriors = []
        for t in range(horizon):
            prev_action = action[t] * nonterms[t]
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embedding[t], prev_action, nonterms[t], prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        
        prior = self.rssm_stack_states(priors, dim=0)
        posterior = self.rssm_stack_states(posteriors, dim=0)

        return prior, posterior

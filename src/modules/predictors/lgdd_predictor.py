import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.th_utils import orthogonal_init_


class TeamActionPredictor(nn.Module):
    """LGDD v3: Cross-agent action prediction (coordination sensor).

    Each agent i uses its own (z_i, g_i) to predict the actions taken by
    every other agent j != i at the same timestep.

    Why actions instead of latent states?
    - Action space is FIXED — action 5 means the same at step 1K and 3M.
      No representation drift, unlike latent-space prediction where the
      encoder reshapes representations every gradient step.
    - Directly measures coordination: "can agent i predict what teammates
      do from its own observation?" This IS implicit coordination.
    - Cross-entropy loss has natural scale: log(n_actions) for random,
      ~0 for perfect prediction. Gives meaningful dynamic range.
    - Signal naturally drops as agents develop coordinated strategies,
      unlike latent prediction which stays flat at a "tracking floor".

    Input per agent:  (z_i, g_i)  — dim = hidden_dim * 2 (NO action input)
    Output per agent: action logits for all n_agents — [n_agents, n_actions]
    Loss: cross-entropy on j != i predictions only (self-prediction masked).
    """

    def __init__(self, hidden_dim, n_actions, n_agents, predictor_hidden_dim=None,
                 use_orthogonal=False, gain=1.0):
        super(TeamActionPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        predictor_hidden_dim = predictor_hidden_dim or hidden_dim

        input_dim = hidden_dim * 2       # (z_i, g_i), NO action
        output_dim = n_agents * n_actions  # predict all agents' actions

        self.fc1 = nn.Linear(input_dim, predictor_hidden_dim)
        self.fc2 = nn.Linear(predictor_hidden_dim, predictor_hidden_dim)
        self.action_head = nn.Linear(predictor_hidden_dim, output_dim)

        if use_orthogonal:
            for module in [self.fc1, self.fc2, self.action_head]:
                orthogonal_init_(module, gain=gain)

    def forward(self, local_summary, team_summary):
        """
        Args:
            local_summary:  [B, n_agents, hidden_dim]
            team_summary:   [B, n_agents, hidden_dim]

        Returns:
            action_logits: [B, n_agents_i, n_agents_j, n_actions]
            Row i contains agent i's predicted action logits for all agents j.
        """
        x = th.cat([local_summary, team_summary], dim=-1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        logits = self.action_head(x)  # [B, n_agents, n_agents * n_actions]
        B = local_summary.size(0)
        return logits.view(B, self.n_agents, self.n_agents, self.n_actions)

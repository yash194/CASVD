import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.th_utils import orthogonal_init_


class LatentDynamicsPredictor(nn.Module):
    """Legacy LGDD predictor for SAC learner.

    Predicts next-step team summary from (local_summary, team_summary, action_onehot).
    Kept for backward compatibility with sac_learner.py.
    """

    def __init__(self, hidden_dim, n_actions, predictor_hidden_dim=None,
                 use_orthogonal=False, gain=1.0):
        super(LatentDynamicsPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        predictor_hidden_dim = predictor_hidden_dim or hidden_dim
        input_dim = hidden_dim * 2 + n_actions
        output_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, predictor_hidden_dim)
        self.fc2 = nn.Linear(predictor_hidden_dim, predictor_hidden_dim)
        self.team_head = nn.Linear(predictor_hidden_dim, output_dim)

        if use_orthogonal:
            for module in [self.fc1, self.fc2, self.team_head]:
                orthogonal_init_(module, gain=gain)

    def forward(self, local_summary, team_summary, action_onehot):
        x = th.cat([local_summary, team_summary, action_onehot], dim=-1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return {
            "pred_team": self.team_head(x),
        }

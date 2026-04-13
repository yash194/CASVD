import copy
import os

import torch as th
import torch.nn.functional as F
from torch.optim import Adam, RMSprop

from components.standarize_stream import RunningMeanStd
from utils.device import move_optimizer_state
from modules.critics import REGISTRY as critic_REGISTRY
from modules.predictors import LatentDynamicsPredictor


class SACLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.target_mac = copy.deepcopy(self.mac)
        self.critic = critic_REGISTRY[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.lgdd_enabled = getattr(args, "lgdd_enabled", False)
        self.lgdd_pretrain_steps = getattr(args, "lgdd_pretrain_steps", 0)
        self.lgdd_continue_after_warmup = getattr(args, "lgdd_continue_after_warmup", True)
        self.lgdd_pretrain_loss_weight = getattr(args, "lgdd_pretrain_loss_weight", 1.0)
        self.lgdd_loss_weight = getattr(args, "lgdd_loss_weight", 0.1)
        self.lgdd_local_loss_weight = getattr(args, "lgdd_local_loss_weight", 1.0)
        self.lgdd_global_loss_weight = getattr(args, "lgdd_global_loss_weight", 1.0)
        self.lgdd_ema_tau = getattr(args, "lgdd_ema_tau", 0.01)
        self.cl_enabled = getattr(args, "cl_enabled", False)
        self.cl_policy_distill_weight = getattr(args, "cl_policy_distill_weight", 0.0)
        self.cl_repr_weight = getattr(args, "cl_repr_weight", 0.0)
        self.cl_teacher_ema_tau = getattr(args, "cl_teacher_ema_tau", 0.002)

        self.agent_params = list(self.mac.parameters())
        self.critic_params = list(self.critic.parameters())
        optimizer = getattr(args, "optimizer", "rmsprop")
        if optimizer == "adam":
            eps = getattr(args, "optimizer_epsilon", 1e-7)
            self.agent_optimiser = Adam(self.agent_params, lr=args.lr, eps=eps)
            self.critic_optimiser = Adam(self.critic_params, lr=args.critic_lr, eps=eps)
        elif optimizer == "rmsprop":
            self.agent_optimiser = RMSprop(
                self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
            )
            self.critic_optimiser = RMSprop(
                self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps
            )
        else:
            raise ValueError("Unknown optimizer {}".format(optimizer))

        self.dynamics_predictor = None
        self.target_encoder = None
        self.cl_teacher_mac = None
        self.dynamics_params = []
        self.dynamics_optimiser = None
        if self.lgdd_enabled:
            predictor_hidden_dim = getattr(args, "lgdd_predictor_hidden_dim", args.hidden_dim)
            lgdd_lr = getattr(args, "lgdd_lr", args.lr)
            self.dynamics_predictor = LatentDynamicsPredictor(
                args.hidden_dim,
                self.n_actions,
                predictor_hidden_dim=predictor_hidden_dim,
                use_orthogonal=getattr(args, "use_orthogonal", False),
                gain=getattr(args, "gain", 1.0),
            )
            self.target_encoder = copy.deepcopy(self.mac.agent)
            self._freeze_module(self.target_encoder)
            self.dynamics_params = list(self.dynamics_predictor.parameters())
            if optimizer == "adam":
                self.dynamics_optimiser = Adam(self.dynamics_params, lr=lgdd_lr, eps=eps)
            else:
                self.dynamics_optimiser = RMSprop(
                    self.dynamics_params,
                    lr=lgdd_lr,
                    alpha=args.optim_alpha,
                    eps=args.optim_eps,
                )
        if self.cl_enabled:
            self.cl_teacher_mac = copy.deepcopy(self.mac)
            self._freeze_mac(self.cl_teacher_mac)

        self.target_entropy = -th.log(th.tensor(1.0 / self.n_actions)).item() * 0.98
        self.log_alpha = th.zeros(1, requires_grad=True)
        self.alpha_optimiser = Adam(
            [self.log_alpha], lr=args.alpha_lr, eps=getattr(args, "optimizer_epsilon", 1e-7)
        )
        self.alpha = self.log_alpha.exp().item()

        device = args.device
        if getattr(args, "standardise_returns", False):
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if getattr(args, "standardise_rewards", False):
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0

    def train(self, batch, t_env, episode_num, current_batch_size=None, memory_batch_size=0):
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.expand(-1, -1, self.n_agents)
        encoder_rl_active = t_env >= self.lgdd_pretrain_steps
        lgdd_active = self.lgdd_enabled and (
            (not encoder_rl_active) or self.lgdd_continue_after_warmup
        )

        agent_type_ids = None
        if "agent_types" in batch.scheme:
            agent_type_ids = batch["agent_types"].long().to(batch.device)

        lgdd_loss = None
        lgdd_metrics = None
        if lgdd_active:
            lgdd_metrics = self._compute_lgdd_loss(batch, mask)
            lgdd_loss = lgdd_metrics["loss"]

        memory_batch = None
        if memory_batch_size and memory_batch_size > 0:
            memory_batch = batch[-memory_batch_size:]

        critic_loss = None
        actor_loss = None
        alpha_loss = None
        avg_entropy = None
        q1_taken = None
        q2_taken = None
        targets = None
        distill_loss = None
        repr_loss = None
        critic_grad_norm = th.tensor(0.0, device=batch.device)
        agent_grad_norm = th.tensor(0.0, device=batch.device)
        dynamics_grad_norm = th.tensor(0.0, device=batch.device)

        rewards = batch["reward"][:, :-1]
        if getattr(self.args, "standardise_rewards", False):
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var + 1e-8)

        rewards = rewards.expand(-1, -1, self.n_agents)

        q1_all, q2_all = self.critic(batch, agent_type_ids=agent_type_ids)
        actions = batch["actions"][:, :-1]
        q1_taken = th.gather(q1_all[:, :-1], dim=3, index=actions).squeeze(3)
        q2_taken = th.gather(q2_all[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            self.target_mac.init_hidden(batch.batch_size)
            target_log_probs = []
            target_probs = []
            for t in range(1, batch.max_seq_length):
                log_probs_t, probs_t = self.target_mac.get_log_probs(batch, t)
                target_log_probs.append(log_probs_t)
                target_probs.append(probs_t)
            target_log_probs = th.stack(target_log_probs, dim=1)
            target_probs = th.stack(target_probs, dim=1)

            target_q1, target_q2 = self.target_critic(batch, agent_type_ids=agent_type_ids)
            target_q_min = th.min(target_q1[:, 1:], target_q2[:, 1:])
            target_v = (target_probs * (target_q_min - self.alpha * target_log_probs)).sum(dim=-1)

            if getattr(self.args, "standardise_returns", False):
                target_v = target_v * th.sqrt(self.ret_ms.var + 1e-8) + self.ret_ms.mean

            targets = rewards + self.args.gamma * (1 - terminated).expand_as(target_v) * target_v
            if getattr(self.args, "standardise_returns", False):
                self.ret_ms.update(targets)
                targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var + 1e-8)

        td_error_1 = (q1_taken - targets.detach()) * mask
        td_error_2 = (q2_taken - targets.detach()) * mask
        critic_loss = (td_error_1.pow(2).sum() + td_error_2.pow(2).sum()) / mask.sum()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        self.mac.init_hidden(batch.batch_size)
        log_probs = []
        probs = []
        for t in range(batch.max_seq_length - 1):
            log_probs_t, probs_t = self.mac.get_log_probs(
                batch, t, detach_encoder=not encoder_rl_active
            )
            log_probs.append(log_probs_t)
            probs.append(probs_t)
        log_probs = th.stack(log_probs, dim=1)
        probs = th.stack(probs, dim=1)

        with th.no_grad():
            q1_pi, q2_pi = self.critic(batch, agent_type_ids=agent_type_ids)
            q_pi_min = th.min(q1_pi[:, :-1], q2_pi[:, :-1])

        actor_loss_per = (probs * (self.alpha * log_probs - q_pi_min)).sum(dim=-1)
        actor_loss = (actor_loss_per * mask).sum() / mask.sum()

        total_actor_loss = actor_loss
        if lgdd_loss is not None:
            lgdd_weight = (
                self.lgdd_loss_weight if encoder_rl_active else self.lgdd_pretrain_loss_weight
            )
            total_actor_loss = total_actor_loss + lgdd_weight * lgdd_loss
        cl_metrics = self._compute_cl_regularizers(memory_batch)
        if cl_metrics is not None:
            distill_loss = cl_metrics["distill_loss"]
            repr_loss = cl_metrics["repr_loss"]
            total_actor_loss = (
                total_actor_loss
                + self.cl_policy_distill_weight * distill_loss
                + self.cl_repr_weight * repr_loss
            )

        self.agent_optimiser.zero_grad()
        if self.dynamics_optimiser is not None:
            self.dynamics_optimiser.zero_grad()
        total_actor_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        if self.dynamics_optimiser is not None:
            dynamics_grad_norm = th.nn.utils.clip_grad_norm_(
                self.dynamics_params, self.args.grad_norm_clip
            )
        self.agent_optimiser.step()
        if self.dynamics_optimiser is not None:
            self.dynamics_optimiser.step()

        with th.no_grad():
            entropy = -(probs * log_probs).sum(dim=-1)
            avg_entropy = (entropy * mask).sum() / mask.sum()

        # Increase alpha when entropy falls below target, decrease it when entropy is above target.
        alpha_loss = (self.log_alpha * (avg_entropy - self.target_entropy).detach())
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()
        self.alpha = self.log_alpha.exp().item()

        tau = self.args.target_update_interval_or_tau
        if tau > 1:
            if (episode_num - self.last_target_update_episode) / tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
        else:
            self._update_targets_soft(tau)

        if lgdd_active:
            self._update_target_encoder()
        if self.cl_teacher_mac is not None:
            self._update_teacher_mac()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = max(mask.sum().item(), 1.0)
            if critic_loss is not None:
                self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            if actor_loss is not None:
                self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            if lgdd_metrics is not None:
                self.logger.log_stat("lgdd_loss", lgdd_metrics["loss"].item(), t_env)
                self.logger.log_stat("lgdd_local_loss", lgdd_metrics["local_loss"].item(), t_env)
                self.logger.log_stat("lgdd_global_loss", lgdd_metrics["global_loss"].item(), t_env)
            self.logger.log_stat("encoder_rl_active", float(encoder_rl_active), t_env)
            self.logger.log_stat("alpha", self.alpha, t_env)
            if avg_entropy is not None:
                self.logger.log_stat("entropy", avg_entropy.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
            if self.dynamics_optimiser is not None:
                self.logger.log_stat("lgdd_grad_norm", dynamics_grad_norm.item(), t_env)
            if distill_loss is not None:
                self.logger.log_stat("cl_distill_loss", distill_loss.item(), t_env)
            if repr_loss is not None:
                self.logger.log_stat("cl_repr_loss", repr_loss.item(), t_env)
            if q1_taken is not None:
                self.logger.log_stat("q1_taken_mean", (q1_taken * mask).sum().item() / mask_elems, t_env)
                self.logger.log_stat("q2_taken_mean", (q2_taken * mask).sum().item() / mask_elems, t_env)
                self.logger.log_stat("target_mean", (targets * mask).sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        device = getattr(self.args, "device", "cuda")
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.to(device)
        self.target_critic.to(device)
        if self.dynamics_predictor is not None:
            self.dynamics_predictor.to(device)
        if self.target_encoder is not None:
            self.target_encoder.to(device)
            self._freeze_module(self.target_encoder)
        if self.cl_teacher_mac is not None:
            self.cl_teacher_mac.cuda()
            self._freeze_mac(self.cl_teacher_mac)
        self.log_alpha = self.log_alpha.to(device)
        self.alpha_optimiser = Adam(
            [self.log_alpha],
            lr=self.args.alpha_lr,
            eps=getattr(self.args, "optimizer_epsilon", 1e-7),
        )

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.log_alpha, "{}/log_alpha.th".format(path))
        if self.dynamics_predictor is not None:
            th.save(self.dynamics_predictor.state_dict(), "{}/dynamics_predictor.th".format(path))
            th.save(self.dynamics_optimiser.state_dict(), "{}/dynamics_opt.th".format(path))
        if self.target_encoder is not None:
            th.save(self.target_encoder.state_dict(), "{}/target_encoder.th".format(path))
        if self.cl_teacher_mac is not None:
            th.save(self.cl_teacher_mac.agent.state_dict(), "{}/teacher_agent.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.log_alpha = th.load("{}/log_alpha.th".format(path), map_location=lambda storage, loc: storage)
        self.log_alpha.requires_grad_(True)
        self.log_alpha = self.log_alpha.to(getattr(self.args, "device", "cpu"))
        self.alpha_optimiser = Adam(
            [self.log_alpha],
            lr=self.args.alpha_lr,
            eps=getattr(self.args, "optimizer_epsilon", 1e-7),
        )
        self.alpha = self.log_alpha.exp().item()
        if self.dynamics_predictor is not None:
            predictor_path = "{}/dynamics_predictor.th".format(path)
            if os.path.exists(predictor_path):
                self.dynamics_predictor.load_state_dict(
                    th.load(predictor_path, map_location=lambda storage, loc: storage)
                )
            dynamics_opt_path = "{}/dynamics_opt.th".format(path)
            if os.path.exists(dynamics_opt_path):
                self.dynamics_optimiser.load_state_dict(
                    th.load(dynamics_opt_path, map_location=lambda storage, loc: storage)
                )
        if self.target_encoder is not None:
            target_encoder_path = "{}/target_encoder.th".format(path)
            if os.path.exists(target_encoder_path):
                self.target_encoder.load_state_dict(
                    th.load(target_encoder_path, map_location=lambda storage, loc: storage)
                )
            else:
                self.target_encoder.load_state_dict(self.mac.agent.state_dict())
            self._freeze_module(self.target_encoder)
        if self.cl_teacher_mac is not None:
            teacher_path = "{}/teacher_agent.th".format(path)
            if os.path.exists(teacher_path):
                self.cl_teacher_mac.agent.load_state_dict(
                    th.load(teacher_path, map_location=lambda storage, loc: storage)
                )
            else:
                self.cl_teacher_mac.load_state(self.mac)
            self._freeze_mac(self.cl_teacher_mac)

        device = getattr(self.args, "device", "cpu")
        move_optimizer_state(self.agent_optimiser, device)
        move_optimizer_state(self.critic_optimiser, device)
        move_optimizer_state(self.alpha_optimiser, device)
        move_optimizer_state(self.dynamics_optimiser, device)

    def _compute_lgdd_loss(self, batch, mask):
        local_losses = []
        global_losses = []
        for t in range(batch.max_seq_length - 1):
            current_inputs = self.mac.build_inputs(batch, t)
            next_inputs = self.mac.build_inputs(batch, t + 1)
            current_latents = self.mac.agent.encode(current_inputs)
            with th.no_grad():
                target_latents = self.target_encoder.encode(next_inputs)
            predictor_out = self.dynamics_predictor(
                current_latents["local_summary"],
                current_latents["team_summary"],
                batch["actions_onehot"][:, t].float(),
            )
            local_loss_t = F.mse_loss(
                predictor_out["pred_local"],
                target_latents["local_summary"],
                reduction="none",
            ).mean(dim=-1)
            global_loss_t = F.mse_loss(
                predictor_out["pred_team"],
                target_latents["team_summary"],
                reduction="none",
            ).mean(dim=-1)
            local_losses.append(local_loss_t)
            global_losses.append(global_loss_t)

        local_losses = th.stack(local_losses, dim=1)
        global_losses = th.stack(global_losses, dim=1)
        denom = mask.sum().clamp(min=1.0)
        local_loss = (local_losses * mask).sum() / denom
        global_loss = (global_losses * mask).sum() / denom
        total_loss = (
            self.lgdd_local_loss_weight * local_loss
            + self.lgdd_global_loss_weight * global_loss
        )
        return {
            "loss": total_loss,
            "local_loss": local_loss,
            "global_loss": global_loss,
        }

    def _update_target_encoder(self):
        for target_param, param in zip(self.target_encoder.parameters(), self.mac.agent.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.lgdd_ema_tau) + param.data * self.lgdd_ema_tau
            )

    def _freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _freeze_mac(self, mac):
        mac.agent.eval()
        for param in mac.agent.parameters():
            param.requires_grad_(False)

    def _compute_cl_regularizers(self, memory_batch):
        if (
            memory_batch is None
            or memory_batch.batch_size == 0
            or self.cl_teacher_mac is None
            or (self.cl_policy_distill_weight <= 0 and self.cl_repr_weight <= 0)
        ):
            return None

        mask = memory_batch["filled"][:, :-1].float()
        terminated = memory_batch["terminated"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.expand(-1, -1, self.n_agents)
        denom = mask.sum().clamp(min=1.0)

        distill_loss = th.tensor(0.0, device=memory_batch.device)
        repr_loss = th.tensor(0.0, device=memory_batch.device)

        if self.cl_policy_distill_weight > 0:
            self.mac.init_hidden(memory_batch.batch_size)
            student_log_probs = []
            for t in range(memory_batch.max_seq_length - 1):
                log_probs_t, _ = self.mac.get_log_probs(memory_batch, t, detach_encoder=False)
                student_log_probs.append(log_probs_t)
            student_log_probs = th.stack(student_log_probs, dim=1)

            with th.no_grad():
                self.cl_teacher_mac.init_hidden(memory_batch.batch_size)
                teacher_log_probs = []
                teacher_probs = []
                for t in range(memory_batch.max_seq_length - 1):
                    log_probs_t, probs_t = self.cl_teacher_mac.get_log_probs(memory_batch, t)
                    teacher_log_probs.append(log_probs_t)
                    teacher_probs.append(probs_t)
                teacher_log_probs = th.stack(teacher_log_probs, dim=1)
                teacher_probs = th.stack(teacher_probs, dim=1)

            kl = teacher_probs * (teacher_log_probs - student_log_probs)
            kl = kl.sum(dim=-1)
            distill_loss = (kl * mask).sum() / denom

        if self.cl_repr_weight > 0:
            repr_local = []
            repr_team = []
            for t in range(memory_batch.max_seq_length - 1):
                student_inputs = self.mac.build_inputs(memory_batch, t)
                student_latents = self.mac.agent.encode(student_inputs)
                with th.no_grad():
                    teacher_inputs = self.cl_teacher_mac.build_inputs(memory_batch, t)
                    teacher_latents = self.cl_teacher_mac.agent.encode(teacher_inputs)
                repr_local.append(
                    F.mse_loss(
                        student_latents["local_summary"],
                        teacher_latents["local_summary"],
                        reduction="none",
                    ).mean(dim=-1)
                )
                repr_team.append(
                    F.mse_loss(
                        student_latents["team_summary"],
                        teacher_latents["team_summary"],
                        reduction="none",
                    ).mean(dim=-1)
                )
            repr_local = th.stack(repr_local, dim=1)
            repr_team = th.stack(repr_team, dim=1)
            repr_loss = ((repr_local + repr_team) * mask).sum() / denom

        return {
            "distill_loss": distill_loss,
            "repr_loss": repr_loss,
        }

    def _update_teacher_mac(self):
        for teacher_param, param in zip(self.cl_teacher_mac.agent.parameters(), self.mac.agent.parameters()):
            teacher_param.data.copy_(
                teacher_param.data * (1.0 - self.cl_teacher_ema_tau)
                + param.data * self.cl_teacher_ema_tau
            )

from copy import deepcopy
from pathlib import Path

import yaml

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from .multiagentenv import MultiAgentEnv


SMACV2_CONFIG_DIR = Path(__file__).parent.parent / "config" / "envs" / "smacv2_configs"


def recursive_dict_update(base, updates):
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_dict_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_scenario(scenario_name, **overrides):
    scenario_path = SMACV2_CONFIG_DIR / "{}.yaml".format(scenario_name)
    if not scenario_path.exists():
        raise FileNotFoundError("Unknown SMACv2 scenario config: {}".format(scenario_path))

    with open(scenario_path, "r") as handle:
        scenario_config = yaml.load(handle, Loader=yaml.FullLoader)

    env_args = deepcopy(scenario_config.get("env_args", {}))
    env_args = recursive_dict_update(env_args, overrides)
    return StarCraftCapabilityEnvWrapper(**env_args)


class SMACv2Wrapper(MultiAgentEnv):
    def __init__(self, map_name, seed=None, **kwargs):
        self.env = load_scenario(map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit

        base_env = self._base_env()
        self.n_agents = getattr(base_env, "n_agents", self.env.get_env_info()["n_agents"])
        self.n_enemies = getattr(base_env, "n_enemies", None)
        self.shield_bits_ally = getattr(base_env, "shield_bits_ally", 0)
        self.shield_bits_enemy = getattr(base_env, "shield_bits_enemy", 0)
        self.unit_type_bits = self._infer_unit_type_bits(base_env)
        self.map_type = getattr(base_env, "map_type", "smacv2")
        self.unit_types = self._get_unit_type_names()

    def _base_env(self):
        return getattr(self.env, "env", self.env)

    def _delegate(self, method_name, default=None):
        for obj in (self.env, self._base_env()):
            if hasattr(obj, method_name):
                return getattr(obj, method_name)()
        return default

    def _infer_unit_type_bits(self, base_env):
        if hasattr(base_env, "unit_type_bits"):
            return base_env.unit_type_bits

        capability_config = getattr(self.env, "capability_config", None)
        if isinstance(capability_config, dict):
            team_gen = capability_config.get("team_gen", {})
            unit_types = team_gen.get("unit_types", [])
            if len(unit_types) > 1:
                return len(unit_types)
        return 0

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        return reward, terminated, info

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        return self.env.get_total_actions()

    def reset(self):
        self.env.reset()

    def get_obs_move_feats_size(self):
        return self._delegate("get_obs_move_feats_size")

    def get_obs_enemy_feats_size(self):
        return self._delegate("get_obs_enemy_feats_size")

    def get_obs_ally_feats_size(self):
        return self._delegate("get_obs_ally_feats_size")

    def get_obs_own_feats_size(self):
        return self._delegate("get_obs_own_feats_size")

    def _get_unit_type_names(self):
        capability_config = getattr(self.env, "capability_config", None)
        if isinstance(capability_config, dict):
            team_gen = capability_config.get("team_gen", {})
            unit_types = team_gen.get("unit_types")
            if unit_types:
                return unit_types
        base_env = self._base_env()
        if hasattr(base_env, "unit_type_ids"):
            return list(base_env.unit_type_ids.keys())
        return ["default"]

    def get_unit_types(self):
        return self.unit_types

    def get_agent_types(self):
        try:
            base_env = self._base_env()
            if hasattr(base_env, "agents"):
                type_to_idx = {name: idx for idx, name in enumerate(self.unit_types)}
                agent_types = []
                agent_items = base_env.agents.items()
                try:
                    agent_items = sorted(agent_items, key=lambda item: item[0])
                except Exception:
                    agent_items = list(agent_items)
                for _, agent in agent_items:
                    unit_type = getattr(agent, "unit_type", None)
                    agent_types.append(type_to_idx.get(unit_type, 0))
                if len(agent_types) == self.n_agents:
                    return agent_types
            if hasattr(base_env, "_get_unit_types"):
                type_to_idx = {name: idx for idx, name in enumerate(self.unit_types)}
                return [type_to_idx.get(unit_type, 0) for unit_type in base_env._get_unit_types()]
        except Exception:
            pass
        return [0] * self.n_agents

    def render(self):
        if hasattr(self.env, "render"):
            self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)

    def save_replay(self):
        if hasattr(self.env, "save_replay"):
            self.env.save_replay()

    def get_env_info(self):
        env_info = self.env.get_env_info()
        env_info["n_agents"] = self.n_agents
        if self.n_enemies is not None:
            env_info["n_enemies"] = self.n_enemies
        env_info["shield_bits_ally"] = self.shield_bits_ally
        env_info["shield_bits_enemy"] = self.shield_bits_enemy
        env_info["unit_type_bits"] = self.unit_type_bits
        env_info["map_type"] = self.map_type
        env_info["unit_types"] = self.unit_types
        env_info["n_unit_types"] = len(self.unit_types)
        for key, value in {
            "obs_move_feats_size": self.get_obs_move_feats_size(),
            "obs_enemy_feats_size": self.get_obs_enemy_feats_size(),
            "obs_ally_feats_size": self.get_obs_ally_feats_size(),
            "obs_own_feats_size": self.get_obs_own_feats_size(),
        }.items():
            if value is not None:
                env_info[key] = value
        return env_info

    def get_stats(self):
        if hasattr(self.env, "get_stats"):
            return self.env.get_stats()
        return {}

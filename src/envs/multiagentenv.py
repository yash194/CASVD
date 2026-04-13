class MultiAgentEnv(object):

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        optional_attrs = [
            "n_enemies",
            "shield_bits_ally",
            "shield_bits_enemy",
            "unit_type_bits",
            "map_type",
        ]
        for attr in optional_attrs:
            if hasattr(self, attr):
                env_info[attr] = getattr(self, attr)
        optional_methods = {
            "obs_move_feats_size": "get_obs_move_feats_size",
            "obs_enemy_feats_size": "get_obs_enemy_feats_size",
            "obs_ally_feats_size": "get_obs_ally_feats_size",
            "obs_own_feats_size": "get_obs_own_feats_size",
            "unit_types": "get_unit_types",
        }
        for key, method_name in optional_methods.items():
            if hasattr(self, method_name):
                env_info[key] = getattr(self, method_name)()
        return env_info

    def get_stats(self):
        return {}

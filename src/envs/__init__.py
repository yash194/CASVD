from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)


def register_smac():
    from .starcraft import StarCraft2Env

    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    REGISTRY["sc2v2"] = partial(env_fn, env=SMACv2Wrapper)


def ensure_env_registered(env_name):
    if env_name == "sc2" and "sc2" not in REGISTRY:
        register_smac()
    elif env_name == "sc2v2" and "sc2v2" not in REGISTRY:
        register_smacv2()

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")

import gym

from ccgm.common.envs.utils import AutoResetWrapper

from .impl.all import MINATAR_STRATEGIES_all  # noqa
from .impl.v0 import MINATAR_STRATEGIES_v0  # noqa
from .impl.v1 import MINATAR_STRATEGIES_v1  # noqa
from .utils import MinAtarStandardObservation


def make_minatar_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = AutoResetWrapper(env)
    env = MinAtarStandardObservation(env)
    return env

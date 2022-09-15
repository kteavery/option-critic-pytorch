from abc import abstractmethod
from .paths import (
    get_intervention_dir,
    get_start_state_path,
)
from typing import *
import gym, json

from toybox.envs.atari.amidar import AmidarEnv

# from deep_rl.component.utils import GymEnvironment, DuplicateEnvironment

# from deep_rl.component.utils import (
#     NoopResetEnv,
#     MaxAndSkipEnv,
#     FireResetEnv,
#     WarpFrame,
#     LifeLostEnv, # copied from all 
# )

class AmidarResetWrapper(gym.Wrapper):
    """Resets amidar environment at the start of every episode to an intervened state."""

    def __init__(
        self,
        tbenv: AmidarEnv,
        intv: int,
        lives: int,
    ):
        super().__init__(tbenv)
        self.env = tbenv
        self.toybox = (
            tbenv.toybox
        )  # Why does this fail when ToyboxBaseEnv has a toybox attribute?
        self.intv = intv  # Intervention number 0 - ?
        self.lives = lives

    def reset(self):
        super().reset()
        return self.on_episode_start()

    @abstractmethod
    def on_episode_start(self):
        """On the start of each episode, set the state to the JSON state according to the intervention."""
        # Get JSON state
        environment = "Amidar"
        if self.intv >= 0:
            with open(
                f"{get_intervention_dir(environment)}/{self.intv}.json",
            ) as f:
                iv_state = json.load(f)

        else:
            with open(get_start_state_path(environment)) as f:
                iv_state = json.load(f)

        iv_state["lives"] = self.lives

        # Set state to the reset state
        self.env.cached_state = iv_state
        self.toybox.write_state_json(iv_state)
        obs = self.env.toybox.get_state()
        return obs


def customAmidarResetWrapper(intv, lives):
    class CustomAmidarResetWrapper(AmidarResetWrapper):
        def __init__(self, env):
            super().__init__(
                env,
                intv=intv,
                lives=lives,
            )
    return CustomAmidarResetWrapper


# class ToyboxEnvironment(GymEnvironment):
#     def __init__(self, name, custom_wrapper, *args, **kwargs):
#         # need these for duplication
#         self._args = args
#         self._kwargs = kwargs
#         # construct the environment
#         # toybox gives 4-channel obs by default, but you can enforce 3-channel with kwargs
#         env = gym.make(name + "NoFrameskip-v4", alpha=False, grayscale=False)
#         self.toybox = env.unwrapped.toybox

#         env = custom_wrapper(env)

#         env = NoopResetEnv(env, noop_max=30)
#         env = MaxAndSkipEnv(env)
#         if "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         env = WarpFrame(env)
#         env = LifeLostEnv(env)
#         # initialize
#         super().__init__(env, *args, **kwargs)
#         self._name = name
#         self._custom_wrapper = custom_wrapper

#     @property
#     def name(self):
#         return self._name

#     def duplicate(self, n):
#         return DuplicateEnvironment(
#             [
#                 ToyboxEnvironment(
#                     self._name, self._custom_wrapper, *self._args, **self._kwargs
#                 )
#                 for _ in range(n)
#             ]
#         )


if __name__ == "__main__":
    pass

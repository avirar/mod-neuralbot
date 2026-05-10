import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


def _run_step(env: gym.Env, action: int) -> tuple:
    result = env.step(action)
    obs, reward, terminated, truncated, info = result
    done = terminated or truncated
    info["TimeLimit.truncated"] = truncated and not terminated
    return obs, reward, done, info


def _run_reset(env: gym.Env, seed: int | None, options: dict | None) -> tuple:
    if seed is not None:
        env.reset(seed=seed)
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
        info = {}
    return obs, info


def _stack_obs(obs_list: list, space: spaces.Space) -> np.ndarray | dict | tuple:
    assert isinstance(obs_list, (list, tuple)), "expected list or tuple of observations"
    assert len(obs_list) > 0, "need observations from at least one environment"
    if isinstance(space, spaces.Dict):
        return {key: np.stack([o[key] for o in obs_list]) for key in space.spaces}
    elif isinstance(space, spaces.Tuple):
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs_list]) for i in range(obs_len))
    else:
        return np.stack(obs_list)


class ThreadedVecEnv(VecEnv):
    """
    A vectorized environment wrapper that uses Python threads to run
    multiple environments in parallel. Optimized for IO-bound environments
    (e.g. TCP-based) where the GIL is released during I/O operations.

    Uses a ThreadPoolExecutor to parallelize step() and reset() calls.

    :param env_fns: List of callables that create Gym environments.
    :param start_method: Ignored; kept for API compatibility.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: str | None = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        self.envs = [fn() for fn in env_fns]
        self.pool = ThreadPoolExecutor(max_workers=n_envs, thread_name_prefix="neuralbot")
        self._futures: list[Future] = []

        observation_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space

        super().__init__(n_envs, observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        assert not self.waiting, "already waiting for a step"
        self._futures = [
            self.pool.submit(_run_step, env, int(a))
            for env, a in zip(self.envs, actions)
        ]
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        assert self.waiting, "must call step_async before step_wait"
        results = []
        for f in self._futures:
            try:
                results.append(f.result(timeout=30))
            except Exception:
                results.append((np.zeros(self.observation_space.shape, dtype=np.float32),
                                0.0, True, {}))
        self._futures = []
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _stack_obs(list(obs), self.observation_space), np.stack(rews), np.stack(dones), list(infos)

    def reset(self) -> VecEnvObs:
        if self.waiting:
            for f in self._futures:
                try:
                    f.result(timeout=5)
                except Exception:
                    pass
            self._futures = []
            self.waiting = False

        futures = [
            self.pool.submit(_run_reset, env, self._seeds[i], self._options[i])
            for i, env in enumerate(self.envs)
        ]
        results = []
        for f in futures:
            try:
                results.append(f.result(timeout=30))
            except Exception:
                results.append((np.zeros(self.observation_space.shape, dtype=np.float32), {}))
        obs, self.reset_infos = zip(*results)
        self.reset_infos = list(self.reset_infos)
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(list(obs), self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for f in self._futures:
                try:
                    f.result(timeout=5)
                except Exception:
                    pass
            self._futures = []
            self.waiting = False
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        self.pool.shutdown(wait=True)
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array`."
            )
            return [None for _ in range(self.num_envs)]
        futures = [self.pool.submit(lambda e: e.render(), env) for env in self.envs]
        return [f.result() for f in futures]

    def has_attr(self, attr_name: str) -> bool:
        return all(hasattr(env, attr_name) for env in self.envs)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        targets = self._get_target_envs(indices)
        return [getattr(env, attr_name) for env in targets]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        targets = self._get_target_envs(indices)
        for env in targets:
            setattr(env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        targets = self._get_target_envs(indices)
        return [getattr(env, method_name)(*method_args, **method_kwargs) for env in targets]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        targets = self._get_target_envs(indices)
        from stable_baselines3.common.env_util import is_wrapped
        return [is_wrapped(env, wrapper_class) for env in targets]

    def _get_target_envs(self, indices: VecEnvIndices) -> list[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

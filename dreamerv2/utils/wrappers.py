import gymnasium as gym
import miniatar
import numpy as np


class GymMiniAtar(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, env_name: str, display_time: int = 50) -> None:
        self.display_time = display_time
        self.env_name = env_name
        self.env = miniatar.Environment(env_name)
        self.minimal_actions = self.env.minimal_action_set()
        h, w, c = self.env.state_shape()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self.env.state().transpose(2, 0, 1)

    def step(self, action_idx):
        """
        Index is the action id, considering only the set of minimal actions.
        """
        action = self.minimal_actions[action_idx]
        reward, done = self.env.act(action)
        self.game_over = done
        return self.env.state().transpose(2, 0, 1), reward, done, {}

    def seed(self, seed: str = "None") -> None:
        self.env = miniatar.Environment(self.env_name, seed=seed)

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            self.env.display_state(self.display_time)
        elif mode == "rgb_array":
            return self.env.state()

    def close(self) -> None:
        if self.env.visualized:
            self.env.close_display()

class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int = 1) -> None:
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action: int) -> tuple:
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env: gym.Env, duration: int) -> None:
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0
    
    def step(self, action: int) -> tuple:
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._step = 0
        return self.env.reset()


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)
    
    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()
    
    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference
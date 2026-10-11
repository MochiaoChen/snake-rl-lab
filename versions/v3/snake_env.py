"""Headless Snake environment. Coordinates are (x, y), y grows downwards."""
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size=10):
        super().__init__()
        if size < 5:
            raise ValueError("Board size must be at least 5")
        self.size = size
        self.action_space = spaces.Discrete(3)  # straight, left, right
        self.observation_space = spaces.Box(0, 1, shape=(11,), dtype=np.float32)
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.size // 2
        self.snake = deque([(c, c), (c - 1, c), (c - 2, c)])
        self.direction = 0
        self.score = self.steps = self.hungry_steps = 0
        self.done = False
        self.food = self._new_food()
        return self._obs(), self._info()

    def _new_food(self):
        occupied = set(self.snake)
        free = [(x, y) for y in range(self.size) for x in range(self.size)
                if (x, y) not in occupied]
        return free[int(self.np_random.integers(len(free)))] if free else None

    def _next(self, direction):
        x, y = self.snake[0]
        dx, dy = DIRECTIONS[direction]
        return x + dx, y + dy

    def _collision(self, point, grows=False):
        x, y = point
        # The tail moves away on a non-eating step: entering it is legal.
        body = list(self.snake) if grows else list(self.snake)[:-1]
        return not (0 <= x < self.size and 0 <= y < self.size) or point in body

    def _obs(self):
        x, y = self.snake[0]
        fx, fy = self.food if self.food is not None else (x, y)
        relative = [self.direction, (self.direction - 1) % 4,
                    (self.direction + 1) % 4]
        danger = [self._collision(self._next(d), self._next(d) == self.food)
                  for d in relative]
        return np.asarray(danger + [self.direction == d for d in range(4)]
                          + [fx < x, fx > x, fy < y, fy > y], dtype=np.float32)

    def _info(self, reason=""):
        return {"score": self.score, "length": len(self.snake),
                "steps": self.steps, "reason": reason}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished game")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0 (straight), 1 (left), or 2 (right)")
        self.direction = (self.direction + (0, -1, 1)[action]) % 4
        point = self._next(self.direction)
        eats = point == self.food
        self.steps += 1
        self.hungry_steps += 1
        if self._collision(point, eats):
            self.done = True
            return self._obs(), -10.0, True, False, self._info("collision")
        self.snake.appendleft(point)
        reward = -0.01
        if eats:
            self.score += 1
            self.hungry_steps = 0
            self.food = self._new_food()
            reward = 10.0
        else:
            self.snake.pop()
        won = self.food is None
        # Hard episode cap also protects evaluations from never-ending policies.
        truncated = not won and (self.hungry_steps >= 100 * len(self.snake)
                                 or self.steps >= 10_000)
        self.done = won or truncated
        reason = "board_full" if won else "time_limit" if truncated else ""
        return self._obs(), reward, won, truncated, self._info(reason)

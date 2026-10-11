"""Fully observed Snake: three board planes plus local topology features."""
from collections import deque
import numpy as np
from gymnasium import spaces
from snake_env import SnakeEnv, DIRECTIONS


class GlobalSnakeEnv(SnakeEnv):
    def __init__(self, size=10):
        super().__init__(size)
        # Body release order, head, food; direction, hunger/clock and topology.
        self.observation_space = spaces.Box(0, 1, shape=(3 * size * size + 24,), dtype=np.float32)

    def action_masks(self):
        if self.food is None:
            return np.ones(3, dtype=bool)
        result = np.asarray([not self._collision(self._next(d), self._next(d) == self.food)
                             for d in (self.direction, (self.direction - 1) % 4,
                                       (self.direction + 1) % 4)], dtype=bool)
        # No legal move means unavoidable death. Allow an action to end the game.
        return result if result.any() else np.ones(3, dtype=bool)

    def _topology(self):
        features = []
        n = self.size * self.size
        for d in (self.direction, (self.direction - 1) % 4, (self.direction + 1) % 4):
            head = self._next(d)
            eats = head == self.food
            if self._collision(head, eats):
                features.extend((0., 0., 1.))
                continue
            body = [head] + list(self.snake)
            if not eats:
                body.pop()
            # Count connected free space, distance to food, and access to the tail.
            blocked = set(body[1:-1])
            tail = body[-1]
            queue = deque([(head, 0)])
            seen = {head}
            distance = 0 if eats else n
            while queue:
                (x, y), dist = queue.popleft()
                for dx, dy in DIRECTIONS:
                    p = (x + dx, y + dy)
                    if (0 <= p[0] < self.size and 0 <= p[1] < self.size
                            and p not in seen and p not in blocked):
                        seen.add(p)
                        queue.append((p, dist + 1))
                        if p == self.food:
                            distance = dist + 1
            features.extend((len(seen) / n, float(tail in seen), min(distance / n, 1.)))
        return features

    def _obs(self):
        n = self.size * self.size
        board = np.zeros((3, self.size, self.size), dtype=np.float32)
        for i, (x, y) in enumerate(self.snake):
            board[0, y, x] = (len(self.snake) - i) / n
        x, y = self.snake[0]
        board[1, y, x] = 1
        if self.food is not None:
            fx, fy = self.food
            board[2, fy, fx] = 1
        direction = [float(self.direction == d) for d in range(4)]
        clock = [min(self.hungry_steps / (100 * len(self.snake)), 1),
                 min(self.steps / 10_000, 1)]
        mask = self.action_masks().astype(np.float32)
        # Last 3 features are immediate legality; the learner uses them as masks.
        local = super()._obs()[7:].tolist()
        extra = direction + clock + local + [len(self.snake) / n, self.score / n] + self._topology()
        return np.concatenate((board.ravel(), np.asarray(extra, dtype=np.float32), mask))

    def step(self, action):
        old_distance = (abs(self.food[0] - self.snake[0][0]) + abs(self.food[1] - self.snake[0][1])) if self.food else 0
        obs, reward, terminated, truncated, info = super().step(action)
        if reward == -0.01 and self.food is not None:
            distance = abs(self.food[0] - self.snake[0][0]) + abs(self.food[1] - self.snake[0][1])
            reward += .05 * (old_distance - distance)
        return obs, reward, terminated, truncated, info

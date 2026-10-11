from collections import deque
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from snake_env import SnakeEnv


def test_api_and_seed():
    env = SnakeEnv()
    check_env(env)
    first = env.reset(seed=9)
    food = env.food
    second = env.reset(seed=9)
    np.testing.assert_array_equal(first[0], second[0])
    assert env.food == food


def test_eating_grows_and_food_stays_off_body():
    env = SnakeEnv()
    env.reset(seed=2)
    env.food = (6, 5)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 10 and info["score"] == 1 and len(env.snake) == 4
    assert env.food not in env.snake and not terminated and not truncated


def test_wall_and_finished_guard():
    env = SnakeEnv()
    env.reset()
    env.snake = deque([(9, 5), (8, 5), (7, 5)])
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and reward == -10
    assert info["reason"] == "collision"
    with pytest.raises(RuntimeError):
        env.step(0)


def test_moving_tail_is_legal_but_body_is_not():
    env = SnakeEnv()
    env.reset()
    env.snake = deque([(2, 2), (2, 3), (1, 3), (1, 2)])
    env.direction = 3
    env.food = (8, 8)
    _, _, terminated, _, _ = env.step(1)
    assert not terminated and env.snake[0] == (1, 2)
    env.reset()
    env.snake = deque([(2, 2), (2, 3), (1, 3), (1, 2)])
    env.direction = 0
    _, _, terminated, _, _ = env.step(2)
    assert terminated


def test_time_limit_is_truncation():
    env = SnakeEnv()
    env.reset()
    env.hungry_steps = 299
    env.food = (0, 0)
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and info["reason"] == "time_limit"


def test_filling_board_wins():
    env = SnakeEnv(5)
    env.reset()
    food, head = (4, 4), (3, 4)
    env.snake = deque([head] + [(x, y) for y in range(5) for x in range(5)
                              if (x, y) not in (head, food)])
    env.food, env.direction = food, 0
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and reward == 10
    assert env.food is None and info["reason"] == "board_full"

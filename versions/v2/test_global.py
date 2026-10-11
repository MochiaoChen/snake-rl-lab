from collections import deque
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from global_env import GlobalSnakeEnv
from global_dqn import GlobalDQN


def test_complete_board_and_tail_order():
    env=GlobalSnakeEnv()
    check_env(env)
    obs,_=env.reset(seed=7)
    assert obs.shape==(324,)
    planes=obs[:300].reshape(3,10,10)
    assert planes[0,5,5]==np.float32(.03)
    assert planes[0,5,4]==np.float32(.02)
    assert planes[0,5,3]==np.float32(.01)
    assert planes[1].sum()==1 and planes[2].sum()==1
    other=GlobalSnakeEnv(); other.reset(seed=7)
    other.snake.append((3,6))
    assert not np.array_equal(obs,other._obs())


def test_mask_respects_wall_and_vacating_tail():
    env=GlobalSnakeEnv();env.reset()
    env.snake=deque([(9,5),(8,5),(7,5)])
    assert not env.action_masks()[0]
    env.snake=deque([(2,2),(2,3),(1,3),(1,2)])
    env.direction=3;env.food=(8,8)
    assert env.action_masks()[1]


def test_mask_applies_to_exploration_and_saved_policy(tmp_path):
    torch.set_num_threads(2)
    env=GlobalSnakeEnv();env.reset()
    env.snake=deque([(9,5),(8,5),(7,5)])
    obs=env._obs()
    model=GlobalDQN('MlpPolicy',env,buffer_size=100,learning_starts=0,batch_size=8,
                    policy_kwargs={'net_arch':[16]},device='cpu')
    model.exploration_rate=1.
    for _ in range(50):assert int(model.predict(obs)[0])!=0
    assert int(model.predict(obs,deterministic=True)[0])!=0
    # Exercise n-step replay, masks and Double-DQN backward together.
    model=GlobalDQN('MlpPolicy',env,buffer_size=100,learning_starts=8,batch_size=8,n_steps=3,
                    policy_kwargs={'net_arch':[16]},device='cpu')
    model.learn(80)
    path=tmp_path/'model';model.save(path)
    loaded=GlobalDQN.load(path,device='cpu')
    obs,_=env.reset(seed=33)
    assert int(model.predict(obs,deterministic=True)[0])==int(loaded.predict(obs,deterministic=True)[0])

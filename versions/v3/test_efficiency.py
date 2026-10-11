from global_env import GlobalSnakeEnv
from train_v3 import EfficiencyTraining


def test_no_food_failure_is_training_only():
    base=GlobalSnakeEnv();train=EfficiencyTraining(base,limit=3)
    train.reset(seed=2)
    base.food=(0,0)
    train.step(0); train.step(0)
    _,reward,terminated,truncated,info=train.step(0)
    assert terminated and not truncated and info['reason']=='training_no_food'
    assert reward < -9
    normal=GlobalSnakeEnv();normal.reset(seed=2);normal.food=(0,0)
    normal.step(0);normal.step(0)
    _,_,terminated,truncated,_=normal.step(0)
    assert not terminated and not truncated

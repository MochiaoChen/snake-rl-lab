"""Refine the v2 policy for efficient play; original game rules used in evaluation."""
import argparse
import json
import time
from pathlib import Path
import gymnasium as gym
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from global_env import GlobalSnakeEnv
from global_dqn import GlobalDQN
from train_v2 import Progress, evaluate


class EfficiencyTraining(gym.Wrapper):
    """Training-only failure signal for spending 400 moves without food."""
    def __init__(self, env, limit=400):
        super().__init__(env)
        self.limit = limit

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Eating/death rewards remain the same. Moving now costs 0.05 instead of 0.01.
        if not terminated and reward < 1:
            reward -= .04
        if not terminated and not truncated and self.env.unwrapped.hungry_steps >= self.limit:
            self.env.unwrapped.done = True
            terminated = True
            reward -= 10.
            info = info | {'reason': 'training_no_food'}
        return obs, reward, terminated, truncated, info


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=250_000)
    p.add_argument('--out',default='runs/v3-efficiency')
    p.add_argument('--source',default='versions/v2/run/best.zip')
    p.add_argument('--eval-episodes',type=int,default=50)
    args=p.parse_args()
    if min(args.steps,args.eval_episodes)<=0:p.error('positive steps and episodes required')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):p.error('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    env=Monitor(EfficiencyTraining(GlobalSnakeEnv()),str(out/'episodes'))
    model=GlobalDQN.load(args.source,env=env,device='cpu')
    model.learning_rate=5e-5
    model._setup_lr_schedule()
    model.exploration_initial_eps=.2
    model.exploration_final_eps=.03
    model.exploration_fraction=.4
    model.exploration_schedule=get_linear_fn(.2,.03,.4)
    config=vars(args)|{'version':'v3','size':10,'algorithm':'GlobalDQN',
        'observation':'same full board as v2','training_only_no_food_limit':400,
        'step_cost':.05,'test_rules':'original 100*length hunger limit and 10000 step cap',
        'test_seed_start':400_000,'source_weights_only':True}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config,indent=2),flush=True)
    start=time.monotonic()
    # Compare to the source on the same validation seeds BEFORE any updates.
    before=evaluate(model,10,seed=50_000)
    model.save(out/'best')
    cb=Progress(out,args.steps,min(25_000,args.steps),10)
    cb.best=before['mean_score']
    (out/'before_validation.json').write_text(json.dumps(before,indent=2))
    interrupted=False
    try:model.learn(args.steps,callback=cb)
    except KeyboardInterrupt:interrupted=True
    model.save(out/'final')
    best=GlobalDQN.load(out/'best',device='cpu')
    result={'actual_timesteps':model.num_timesteps,'interrupted':interrupted,
            'before_validation':before,'best_validation_mean':cb.best,
            'best':evaluate(best,args.eval_episodes,seed=400_000)}
    result['elapsed_seconds']=round(time.monotonic()-start,2)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    (out/'progress.json').write_text(json.dumps({'status':'interrupted' if interrupted else 'completed',
        'timesteps':model.num_timesteps,'total_steps':args.steps,'elapsed_seconds':result['elapsed_seconds'],
        'best':result['best']},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([0]+[r['timesteps'] for r in cb.rows],[before['mean_score']]+[r['mean_score'] for r in cb.rows],marker='o')
    ax.set(xlabel='Additional training steps',ylabel='Mean food / game (10 validation games)',
           title='v3: efficiency refinement');ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'learning_curve.png',dpi=160);plt.close(fig)
    env.close()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

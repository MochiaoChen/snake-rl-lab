"""Train the full-board v2 Double DQN without route planning."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from global_env import GlobalSnakeEnv
from global_dqn import GlobalDQN


def evaluate(model=None, episodes=50, size=10, seed=300_000):
    env = GlobalSnakeEnv(size)
    scores, steps, reasons = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed+i)
        rng = np.random.default_rng(seed+i)
        while True:
            action = int(rng.choice(np.flatnonzero(env.action_masks()))) if model is None else int(
                model.predict(obs, deterministic=True)[0])
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                scores.append(info['score']); steps.append(info['steps']); reasons.append(info['reason'])
                break
    return {'episodes': episodes, 'seed_start': seed, 'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)), 'min_score': min(scores), 'max_score': max(scores),
            'mean_steps': float(np.mean(steps)), 'scores': scores,
            'board_full': reasons.count('board_full'), 'collision': reasons.count('collision'),
            'time_limit': reasons.count('time_limit')}


class Progress(BaseCallback):
    def __init__(self, out, total, interval, episodes):
        super().__init__()
        self.out, self.total, self.interval, self.episodes = out, total, interval, episodes
        self.rows, self.best = [], -1.
        self.start = time.monotonic()

    def _on_step(self):
        if self.num_timesteps % self.interval != 0:
            return True
        stats = evaluate(self.model, self.episodes, seed=50_000)
        elapsed = time.monotonic() - self.start
        row = {'timesteps': self.num_timesteps, 'mean_score': stats['mean_score'],
               'max_score': stats['max_score'], 'mean_steps': stats['mean_steps'],
               'elapsed_seconds': round(elapsed, 1)}
        self.rows.append(row)
        with (self.out/'eval.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(self.rows)
        progress=row | {'total_steps': self.total, 'percent': round(100*self.num_timesteps/self.total,1),
                        'eta_seconds': round(elapsed/self.num_timesteps*(self.total-self.num_timesteps))}
        (self.out/'progress.json').write_text(json.dumps(progress,indent=2))
        self.model.save(self.out/'latest')
        if stats['mean_score'] > self.best:
            self.best=stats['mean_score'];self.model.save(self.out/'best')
        print(f"{progress['percent']:5.1f}% | {self.num_timesteps:,}/{self.total:,} | "
              f"mean food {stats['mean_score']:.1f}, max {stats['max_score']} | "
              f"elapsed {elapsed:.0f}s, ETA {progress['eta_seconds']}s", flush=True)
        return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=500_000)
    p.add_argument('--out',default='runs/v2-global')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-episodes',type=int,default=50)
    p.add_argument('--resume')
    p.add_argument('--interval',type=int,default=25_000)
    p.add_argument('--validation-episodes',type=int,default=5)
    args=p.parse_args()
    if min(args.steps,args.eval_episodes,args.interval,args.validation_episodes)<=0:
        p.error('steps, interval and episode counts must be positive')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):p.error('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    env=Monitor(GlobalSnakeEnv(),str(out/'episodes'))
    model=GlobalDQN('MlpPolicy',env,device='cpu',seed=args.seed,learning_rate=1e-4,
                    buffer_size=100_000,learning_starts=2000,batch_size=128,
                    train_freq=4,gradient_steps=1,target_update_interval=2000,
                    gamma=.99,n_steps=3,exploration_fraction=.5,exploration_final_eps=.05,
                    policy_kwargs={'net_arch':[256,128]},verbose=0)
    if args.resume:model=GlobalDQN.load(args.resume,env=env,device='cpu')
    config=vars(args)|{'version':'v2','size':10,'algorithm':'GlobalDQN',
                       'observation':'full body order + head + food + direction + clocks + topology',
                       'action_mask':'immediate collision only; no route planner',
                       'trainable_parameters':sum(p.numel() for p in model.q_net.parameters())}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config,indent=2),flush=True)
    start=time.monotonic()
    cb=Progress(out,args.steps,min(args.interval,args.steps),args.validation_episodes)
    interrupted=False
    try:model.learn(args.steps,callback=cb)
    except KeyboardInterrupt:interrupted=True
    model.save(out/'final')
    if not (out/'best.zip').exists():model.save(out/'best')
    best=GlobalDQN.load(out/'best',device='cpu')
    result={'actual_timesteps':model.num_timesteps,'interrupted':interrupted,
            'best':evaluate(best,args.eval_episodes)}
    result['elapsed_seconds']=round(time.monotonic()-start,2)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    (out/'progress.json').write_text(json.dumps({'status':'interrupted' if interrupted else 'completed',
        'timesteps':model.num_timesteps,'total_steps':args.steps,'elapsed_seconds':result['elapsed_seconds'],
        'best':result['best']},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([r['timesteps'] for r in cb.rows],[r['mean_score'] for r in cb.rows],marker='o')
    ax.set(xlabel='Training steps',ylabel=f'Mean food / game ({args.validation_episodes} validation games)',
           title='v2: full-board Double DQN');ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'learning_curve.png',dpi=160);plt.close(fig)
    env.close()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

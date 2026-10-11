"""Compare archived v1, v1 with the same collision guard, and full-board v2."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import DQN
from train import evaluate as evaluate_v1
from train_v2 import evaluate as evaluate_v2
from global_dqn import GlobalDQN


class GuardedV1:
    def __init__(self, model):self.model=model
    def predict(self, obs, deterministic=True):
        valid=np.asarray(obs[:3]) < .5
        if not valid.any():valid=np.ones(3,dtype=bool)
        with torch.no_grad():
            tensor,_=self.model.policy.obs_to_tensor(obs)
            q=self.model.q_net(tensor)[0].cpu().numpy()
        return int(np.where(valid,q,-1e9).argmax()),None


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--v2',default='versions/v2/run/best.zip')
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--seed',type=int,default=300_000)
    p.add_argument('--out',default='runs/v2-global/comparison.json')
    args=p.parse_args()
    torch.set_num_threads(2)
    old=DQN.load('versions/v1/run/best.zip',device='cpu')
    new=GlobalDQN.load(args.v2,device='cpu')
    result={'seed_start':args.seed,'episodes':args.episodes,
            'v1':evaluate_v1(old,args.episodes,seed=args.seed),
            'v1_with_collision_guard':evaluate_v1(GuardedV1(old),args.episodes,seed=args.seed),
            'v2':evaluate_v2(new,args.episodes,seed=args.seed)}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

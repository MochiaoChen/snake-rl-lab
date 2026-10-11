"""Evaluate the active model on fresh seeds against a random baseline."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import DQN
from train import evaluate as evaluate_v1
from train_v2 import evaluate as evaluate_v2
from global_dqn import GlobalDQN

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed',type=int,default=200_000)
    args=p.parse_args()
    if args.episodes<=0:p.error('episodes must be positive')
    active=json.loads(Path('active_model.json').read_text())
    path=Path(args.model or active['model'])
    config=json.loads((path.parent/'config.json').read_text())
    global_version=config.get('algorithm')=='GlobalDQN'
    Loader=GlobalDQN if global_version else DQN
    evaluate=evaluate_v2 if global_version else evaluate_v1
    torch.set_num_threads(2)
    model=Loader.load(path,device='cpu')
    result={'random':evaluate(None,args.episodes,config['size'],args.seed),
            'model':evaluate(model,args.episodes,config['size'],args.seed),
            'random_collision_mask':global_version}
    print(json.dumps(result,indent=2))

"""Evaluate a saved policy against a seeded random baseline."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import DQN
from train import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/first/best.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=200_000)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    torch.set_num_threads(2)
    config = json.loads((Path(args.model).parent / "config.json").read_text())
    model = DQN.load(args.model, device="cpu")
    result = {"random": evaluate(None, args.episodes, config["size"], args.seed),
              "model": evaluate(model, args.episodes, config["size"], args.seed)}
    print(json.dumps(result, indent=2))

"""Train DQN, save checkpoints, and evaluate on fixed held-out seeds."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeEnv


def evaluate(model=None, episodes=50, size=10, seed=100_000):
    env = SnakeEnv(size)
    scores, lengths, rewards = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        rng = np.random.default_rng(seed + i)
        total = 0.0
        while True:
            action = int(rng.integers(3)) if model is None else int(
                model.predict(obs, deterministic=True)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                scores.append(info["score"])
                lengths.append(info["steps"])
                rewards.append(total)
                break
    env.close()
    return {"episodes": episodes, "seed_start": seed,
            "mean_score": float(np.mean(scores)), "max_score": max(scores),
            "mean_steps": float(np.mean(lengths)),
            "mean_reward": float(np.mean(rewards)), "scores": scores}


class Progress(BaseCallback):
    def __init__(self, directory, size, interval=10_000):
        super().__init__()
        self.directory, self.size, self.interval = directory, size, interval
        self.best = -float("inf")
        self.rows = []

    def _on_step(self):
        if self.num_timesteps % self.interval == 0:
            result = evaluate(self.model, 20, self.size, seed=50_000)
            row = {"timesteps": self.num_timesteps,
                   "mean_score": result["mean_score"],
                   "max_score": result["max_score"]}
            self.rows.append(row)
            with (self.directory / "eval.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(self.rows)
            self.model.save(self.directory / "latest")
            if result["mean_score"] > self.best:
                self.best = result["mean_score"]
                self.model.save(self.directory / "best")
            print(f"Evaluation {self.num_timesteps:,}: mean food "
                  f"{result['mean_score']:.2f}, max {result['max_score']}", flush=True)
        return True


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((directory / "eval.csv").open()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([int(r["timesteps"]) for r in rows],
            [float(r["mean_score"]) for r in rows], marker="o")
    ax.set(xlabel="Training steps", ylabel="Mean food / game (20 eval games)",
           title="Snake DQN — held-out checkpoint evaluation")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / "learning_curve.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--out", default="runs/first")
    parser.add_argument("--resume", help="Path to a saved model zip")
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_episodes <= 0:
        parser.error("steps and eval-episodes must be positive")
    directory = Path(args.out)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Output directory is not empty; choose a new --out to preserve results")
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    env = Monitor(SnakeEnv(args.size), str(directory / "episodes"))
    model = DQN("MlpPolicy", env, device="cpu", seed=args.seed,
                learning_rate=5e-4, buffer_size=50_000, learning_starts=1_000,
                batch_size=64, train_freq=4, gradient_steps=1,
                target_update_interval=1_000, gamma=.95,
                exploration_fraction=.4, exploration_final_eps=.05,
                policy_kwargs={"net_arch": [128, 128]}, verbose=0)
    if args.resume:
        model = DQN.load(args.resume, env=env, device="cpu")
        # Resume the learned weights; collect fresh experience in this new run.
    config = vars(args) | {"device": "cpu", "torch_threads": 2,
                           "observation": "11 local features",
                           "trainable_parameters": sum(p.numel() for p in model.q_net.parameters())}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    baseline = evaluate(None, args.eval_episodes, args.size)
    untrained = evaluate(model, args.eval_episodes, args.size)
    callback = Progress(directory, args.size, min(10_000, args.steps))
    start = time.monotonic()
    interrupted = False
    try:
        model.learn(args.steps, callback=callback)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving completed updates.", flush=True)
    model.save(directory / "final")
    model.save_replay_buffer(directory / "replay_buffer.pkl")
    if not (directory / "best.zip").exists():
        model.save(directory / "best")
    best = DQN.load(directory / "best", device="cpu")
    result = {"elapsed_seconds": round(time.monotonic() - start, 2),
              "actual_timesteps": model.num_timesteps, "interrupted": interrupted,
              "random": baseline, "before_training": untrained,
              "final": evaluate(model, args.eval_episodes, args.size),
              "best": evaluate(best, args.eval_episodes, args.size)}
    (directory / "results.json").write_text(json.dumps(result, indent=2))
    if callback.rows:
        plot(directory)
    env.close()
    print(json.dumps({k: {"mean_score": v["mean_score"], "max_score": v["max_score"]}
                      for k, v in result.items() if isinstance(v, dict)}, indent=2), flush=True)
    print(f"Saved to {directory.resolve()}", flush=True)


if __name__ == "__main__":
    main()

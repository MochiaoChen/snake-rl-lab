# 贪吃蛇 RL 实验室

在 Apple Silicon Mac 上用小型 DQN 学习贪吃蛇。CPU 训练，10×10 棋盘；无需下载大模型。

## 开始

在终端执行：

```sh
cd ~/Documents/lab/snake-rl
uv sync
uv run python play.py
```

也可以双击 `玩贪吃蛇.command`。默认每秒 4 格；窗口打开或重开时等待方向键 / 空格开始。右侧 − / + 按钮或键盘 − / + 可以把速度调到每秒 1–20 格。方向键控制，A 切换 AI（需已训练模型），空格开始 / 暂停，R 重开，Esc 退出。也支持鼠标点击开始、重开和 AI 接管。AI 死亡后按 R 重开，方便查看单局表现。

双击 `看AI玩.command` 加载 `runs/first/best.zip`；双击 `开始训练.command` 创建带时间戳的新实验目录。后续模型可通过 `--model` 指定。

## 训练与观看

```sh
uv run python train.py --steps 150000 --out runs/first
uv run python play.py --ai --model runs/first/best.zip
```

训练不打开游戏画面。每 1 万步使用固定的 20 个验证种子评估并保存最佳模型；结束后使用另外 50 个种子，对比随机策略、训练前模型、最终模型和最佳模型。只看训练奖励上升不足以证明学会了游戏。

`runs/first/` 保存：

- `best.zip`：验证平均得分最高的模型。
- `latest.zip` / `final.zip`：最近检查点 / 最终模型。
- `results.json`：独立测试成绩、实际训练步数和耗时。
- `eval.csv` / `learning_curve.png`：验证曲线。
- `episodes.monitor.csv`：训练对局记录。
- `config.json`：参数、种子和网络大小。
- `replay_buffer.pkl`：训练结束时的经验回放。

输出目录必须为空，避免覆盖旧实验。Ctrl+C 会保存当前模型并做评估。继续训练使用新目录：

```sh
uv run python train.py --steps 150000 --resume runs/first/final.zip --out runs/second
```

此命令恢复网络及优化器，重新收集经验，沿用已保存模型的探索参数；不自动恢复旧经验回放。不同实验只用相同测试种子作对比，避免反复拿测试结果挑模型。

## 模型怎样学习

输入 11 个特征：前/左/右危险、当前四个方向、食物四个相对方向；输出直走、左转、右转。网络为两层、每层 128 单元。吃到食物 +10，死亡 −10，每步 −0.01；长时间没吃到食物或超过 10000 步就截断本局。进入即将移走的尾巴位置是合法的。

这个观察方式不包含完整蛇身，存在部分可观测性：适合入门，长蛇时可能困住自己。后续可升级为完整棋盘输入、小型 CNN，再与 PPO 比较。第一版不保证填满棋盘。

## 独立评估

```sh
uv run python evaluate.py --model runs/first/best.zip --episodes 100 --seed 200000
```

## 检查

```sh
uv run pytest -q
uv run python play.py --ai --model runs/first/best.zip --screenshot runs/first/preview.png
```

依赖由 `uv.lock` 固定。算法实现来自 Stable-Baselines3：https://stable-baselines3.readthedocs.io/en/stable/modules/dqn.html

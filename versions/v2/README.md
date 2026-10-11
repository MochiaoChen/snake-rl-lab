# 贪吃蛇 RL 实验室

在 24GB Apple Silicon Mac 上训练和观看贪吃蛇 AI。项目保留完整的版本过程，见 `HISTORY.md`。

## 游玩

双击 `玩贪吃蛇.command`，或者在终端执行：

```sh
cd ~/Documents/lab/snake-rl
uv sync
uv run python play.py
```

浅色中文界面，默认每秒 4 格，打开或重开等待方向键 / 空格开始。右侧 − / + 或键盘 − / + 可调每秒 1–20 格。空格暂停，R 重开，A 切换 AI，Esc 退出。AI 死亡后按 R 重开。

双击 `看AI玩.command` 默认使用 `active_model.json` 选定的模型。可以显式选择版本：

```sh
uv run python play.py --ai --model versions/v1/run/best.zip
uv run python play.py --ai --model runs/v2-global/best.zip
```

界面右下角显示当前模型版本。调速度只改变两步之间的等待时间，不改变棋盘、模型输入和奖励。

## v2：全局观察训练

```sh
uv run python train_v2.py --steps 500000 --out runs/v2-global
```

输出目录必须为空，旧实验不会被覆盖。双击 `开始训练.command` 自动创建带时间戳的新实验目录。训练保存检查点、完整参数、训练对局、验证曲线、`progress.json` 和独立测试结果；终端显示百分比、验证得分、耗时和 ETA。Ctrl+C 保存已完成的训练。

v2 输入完整棋盘：蛇身位置及顺序、蛇头、食物；补充朝向、时钟和可达空间信息。小型全连接网络 + Double DQN + 3 步回报。行动掩码只禁止下一步立即碰撞，不保证长远安全。没有固定路线和路径规划器。

每 25000 步用 5 个固定验证种子挑选最佳检查点；结束后另外 50 个种子做最终测试。reward 增加了微小的食物距离提示，跨版本比较请使用实际食物数量而非累计 reward。

```sh
uv run python compare_versions.py
```

比较 v1、仅增加同样碰撞检查的 v1、全局 v2；使用同一组食物随机种子。这样能看到收益是否只是来自避开立即碰撞。

## 版本归档

- `versions/v1/`：原始局部观察版代码、锁定依赖、模型和训练结果。保持不变。
- `versions/v2/`：全局观察版的最终代码、模型和验证结果快照。
- `runs/`：每次独立实验的完整记录。
- `HISTORY.md`：需求、实现选择、实验过程、实测成绩与限制。
- `active_model.json`：桌面游玩默认加载哪个版本。

v1 仍可重新训练：`uv run python train.py --steps 150000 --out runs/v1-new`。

## 检查

```sh
uv run pytest -q
uv run python play.py --ai --screenshot runs/preview.png
```

依赖由 `uv.lock` 固定。算法基础来自 Stable-Baselines3：https://stable-baselines3.readthedocs.io/en/stable/modules/dqn.html

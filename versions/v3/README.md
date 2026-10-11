# v3 效率实验

未替换默认 v2。`run/final.zip` 是新候选；`run/best.zip` 因验证集未提升而保留原 v2 权重。

实测取舍与种子见 [过程](HISTORY.md) 和 [原始对比](run/comparison.json)。从仓库根目录观看候选：

```sh
uv run python play.py --ai --model versions/v3/run/final.zip
```

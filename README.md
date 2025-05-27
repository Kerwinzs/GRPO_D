# ddpo-pytorch-main

本项目基于 DDPO-PyTorch，实现了 PPO 到 GRPO 的算法迁移。

## 主要特性
- 支持组级别优化（GRPO）
- 支持参考模型同步与混合
- 详细的训练与评估指标

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 运行训练：`python scripts/train.py --config config/base.py`

## 详细文档
请参考 `GRPO_MODIFICATIONS.md`。

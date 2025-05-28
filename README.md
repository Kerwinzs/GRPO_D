# GRPO-Diffusion: 扩散模型的组级别强化学习训练框架

## 作者
- 臧真硕
- 蒋晨昂
- 黄旺邦

## 项目简介
GRPO-Diffusion是一个基于PyTorch实现的扩散模型强化学习训练框架，通过组级别的策略优化来提升扩散模型的生成质量。本项目提供了一套完整的训练流程，包括采样、奖励计算、策略优化等功能。

## 主要功能

### 1. 组级别优化
- 支持组内样本比较和优化
- 灵活的组大小配置
- 组内奖励归一化

### 2. 多样化的奖励函数
- JPEG压缩性评估
- 美学评分
- 可扩展的奖励函数接口

### 3. 训练优化
- 混合精度训练
- 梯度累积
- 内存优化
- 自动检查点保存

## 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+
- 其他依赖见 `requirements.txt`

## 快速开始

### 1. 安装
```bash
# 克隆仓库
git clone https://github.com/your-username/GRPO-Diffusion.git
cd GRPO-Diffusion

# 创建并激活虚拟环境
conda create -n ddpo-torch python=3.10
conda activate ddpo-torch

# 安装依赖
pip install -e .
```

### 2. 训练模型
```bash
python scripts/train.py --config config/base.py
```

## 项目结构
```
GRPO-Diffusion/
├── config/                 # 配置文件
├── ddpo_pytorch/          # 核心代码
│   ├── rewards.py         # 奖励函数实现
│   ├── stat_tracking.py   # 统计跟踪
│   ├── prompts.py         # 提示词生成
│   └── aesthetic_scorer.py # 美学评分
├── scripts/               # 训练脚本
└── requirements.txt       # 项目依赖
```

## 使用示例

### 基本训练
```python
# 使用默认配置训练
python scripts/train.py --config config/base.py

# 使用自定义配置训练
python scripts/train.py --config config/custom.py
```

### 恢复训练
```python
# 从检查点恢复训练
python scripts/train.py --config config/base.py --resume_from logs/checkpoint_50
```

## 注意事项
1. 确保有足够的GPU内存（建议16GB以上）
2. 训练前检查配置文件参数
3. 定期保存检查点
4. 监控训练日志

## 贡献指南
欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证
本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式
如有问题，请通过Issue或Pull Request进行交流。

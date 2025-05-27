import numpy as np
from collections import deque, defaultdict


class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = defaultdict(lambda: {"rewards": [], "count": 0})
        
    def update(self, prompts, rewards):
        """
        更新统计信息并计算组内相对优势
        
        参数:
        - prompts: 提示词列表
        - rewards: 奖励列表，形状为 (batch_size * group_size,)
        
        返回:
        - advantages: 组内相对优势，形状与 rewards 相同
        """
        # 将奖励重塑为组形式
        rewards = rewards.reshape(-1, self.group_size)
        
        # 计算组内相对优势
        group_means = rewards.mean(axis=1, keepdims=True)
        group_stds = rewards.std(axis=1, keepdims=True) + 1e-8
        advantages = (rewards - group_means) / group_stds
        
        # 更新每个提示词的统计信息
        for prompt, reward in zip(prompts, rewards.flatten()):
            self.stats[prompt]["rewards"].append(reward)
            self.stats[prompt]["count"] += 1
            
            # 如果超过缓冲区大小，移除最旧的奖励
            if len(self.stats[prompt]["rewards"]) > self.buffer_size:
                self.stats[prompt]["rewards"].pop(0)
        
        return advantages.flatten()
    
    def get_stats(self, prompt):
        """
        获取指定提示词的统计信息
        
        参数:
        - prompt: 提示词
        
        返回:
        - mean: 平均奖励
        - std: 奖励标准差
        """
        if prompt not in self.stats or self.stats[prompt]["count"] < self.min_count:
            return 0.0, 1.0
            
        rewards = self.stats[prompt]["rewards"]
        return np.mean(rewards), np.std(rewards) + 1e-8

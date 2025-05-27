# PPO 到 GRPO 的修改文档

## 1. 配置文件修改 (`config/base.py`)

添加了以下 GRPO 相关的配置参数：
- `train.group_size`: 每组生成的补全数量
- `train.beta`: KL 散度系数
- `train.epsilon`: 裁剪范围
- `train.scale_rewards`: 是否对奖励进行标准化
- `train.use_reference_model`: 是否使用参考模型
- `train.reference_model_mixup_alpha`: 参考模型混合系数
- `train.reference_update_freq`: 参考模型更新频率

## 2. 统计跟踪修改 (`ddpo_pytorch/stat_tracking.py`)

修改了 `PerPromptStatTracker` 类：
- 添加了 `group_stats` 字典来存储组级别的统计信息
- 修改了 `update` 方法以支持组级别的优势计算
- 扩展了 `get_stats` 方法以包含组统计信息

## 3. 训练脚本修改 (`scripts/train.py`)

### 3.1 添加 GRPO 损失函数
```python
def compute_grpo_loss(log_prob, old_log_prob, advantages, beta=0.04, epsilon=0.2):
    # 计算策略比率
    ratio = torch.exp(log_prob - old_log_prob)
    
    # 计算 KL 散度
    kl_div = torch.mean(old_log_prob - log_prob)
    
    # 计算裁剪后的比率
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - epsilon,
        1.0 + epsilon
    )
    
    # 计算策略损失
    policy_loss = -torch.mean(
        torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        )
    )
    
    # 计算总损失
    loss = policy_loss + beta * kl_div
    
    return loss, kl_div, torch.mean(ratio)
```

### 3.2 修改采样逻辑
- 实现了组级别的样本生成
- 添加了组内相对优势计算
- 修改了样本数据结构

### 3.3 修改训练循环
- 实现了 GRPO 损失计算
- 添加了参考模型支持
- 添加了新的评估指标

## 4. 主要变化

1. **组级别优化**：
   - 每个提示词生成多个样本
   - 计算组内相对优势
   - 使用组级别的统计信息

2. **参考模型支持**：
   - 添加了参考模型初始化
   - 实现了定期更新机制
   - 使用混合系数平滑更新

3. **损失函数改进**：
   - 添加了 KL 散度约束
   - 使用组级别的优势计算
   - 实现了更稳定的策略更新

4. **评估指标扩展**：
   - 添加了组级别的奖励统计
   - 添加了 KL 散度监控
   - 添加了参考模型同步状态

## 5. 测试步骤

1. 配置文件测试：
   - 验证所有新添加的配置参数是否正确加载
   - 检查默认值是否合理

2. 采样逻辑测试：
   - 验证组大小样本生成是否正常
   - 检查组内相对优势计算是否正确
   - 确认样本数据结构是否完整

3. 训练逻辑测试：
   - 验证 GRPO 损失计算是否正确
   - 检查 KL 散度计算是否准确
   - 确认优势计算是否符合预期

4. 参考模型测试：
   - 验证参考模型初始化是否成功
   - 检查参考模型更新是否正常
   - 确认混合系数是否生效

5. 评估指标测试：
   - 验证新添加的指标是否正确记录
   - 检查日志输出是否完整
   - 确认指标计算是否准确

## 6. 注意事项

1. 内存使用：
   - 组级别采样会增加内存使用
   - 需要适当调整 batch size 和 group size

2. 训练稳定性：
   - 使用参考模型可以提高训练稳定性
   - 需要仔细调整 beta 和 epsilon 参数

3. 性能监控：
   - 密切关注组级别的奖励变化
   - 监控 KL 散度是否在合理范围内
   - 观察参考模型的更新效果

## 7. 后续优化建议

1. 动态组大小：
   - 根据训练进度动态调整组大小
   - 实现自适应采样策略

2. 参考模型改进：
   - 实现更复杂的参考模型更新策略
   - 添加参考模型质量评估

3. 损失函数优化：
   - 添加额外的正则化项
   - 实现动态 beta 调整

4. 评估指标扩展：
   - 添加更多组级别的统计信息
   - 实现更详细的训练监控 
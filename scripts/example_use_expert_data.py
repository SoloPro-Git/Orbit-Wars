"""示例：如何在训练中使用专家数据。

展示如何加载专家数据并用于监督学习预训练。
"""

import numpy as np
from training.expert import load_expert_dataset
from training.core.feature_engineering import FeatureEngineer


def simple_behavior_cloning_demo():
    """简单的行为克隆示例。"""
    print("=" * 60)
    print("专家数据行为克隆示例")
    print("=" * 60)

    # 1. 加载专家数据
    print("\n1. 加载专家数据...")
    dataset = load_expert_dataset("data/test_expert_demonstrations")
    print(f"   ✓ 加载了 {len(dataset)} 个样本")

    # 2. 分割数据集
    print("\n2. 分割训练集和验证集...")
    train_ds, val_ds = dataset.split(train_ratio=0.8)
    print(f"   ✓ 训练集: {len(train_ds)} 样本")
    print(f"   ✓ 验证集: {len(val_ds)} 样本")

    # 3. 初始化特征工程
    feature_engineer = FeatureEngineer()

    # 4. 处理一个样本
    print("\n3. 示例：处理单个样本...")
    sample = train_ds[0]

    # 提取特征
    features = feature_engineer.extract_features(
        observation=sample["observation"],
        player_id=sample["player_id"],
    )
    print(f"   ✓ 特征形状: {features.shape}")

    # 提取动作（用于监督学习）
    actions = sample["actions"]
    print(f"   ✓ 动作数量: {len(actions)}")
    if actions:
        print(f"   ✓ 示例动作: {actions[0]}")

    # 5. 批量处理
    print("\n4. 批量处理数据...")
    batch = train_ds.get_batch(batch_size=4)
    print(f"   ✓ 获取了 {len(batch)} 个样本")

    batch_features = []
    batch_actions = []

    for sample in batch:
        features = feature_engineer.extract_features(
            observation=sample["observation"],
            player_id=sample["player_id"],
        )
        batch_features.append(features)

        # 将动作转换为模型需要的格式
        # 这里简化处理，实际需要根据你的模型架构调整
        actions = sample["actions"]
        batch_actions.append(actions)

    batch_features = np.array(batch_features)
    print(f"   ✓ 批次特征形状: {batch_features.shape}")

    # 6. 训练循环示例（伪代码）
    print("\n5. 训练循环示例（伪代码）...")
    print("""
    # 初始化模型和优化器
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 提取特征和动作
            features = extract_features(batch)
            target_actions = batch["actions"]

            # 前向传播
            pred_actions = model(features)

            # 计算损失
            loss = compute_loss(pred_actions, target_actions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 实际使用时，可以用这个预训练模型作为起点
    # 然后用 PPO 或其他 RL 算法继续训练
    """)

    print("\n" + "=" * 60)
    print("✓ 示例完成")
    print("=" * 60)


def integration_with_training():
    """展示如何与现有训练代码集成。"""
    print("\n" + "=" * 60)
    print("与现有训练代码集成")
    print("=" * 60)

    print("""
    1. 使用专家数据预训练模型:

    # 在训练脚本中添加预训练阶段
    def pretrain_with_expert_data(model, expert_data_path, num_epochs=10):
        dataset = load_expert_dataset(expert_data_path)
        train_ds, val_ds = dataset.split(train_ratio=0.8)

        for epoch in range(num_epochs):
            for batch in train_ds.get_batch(batch_size=32):
                # 提取特征
                features = extract_features(batch)

                # 监督学习
                loss = supervised_loss(model, features, batch["actions"])
                loss.backward()
                optimizer.step()

    2. 使用预训练模型初始化 RL 训练:

    # 加载预训练模型
    model = load_pretrained_model("checkpoints/pretrained_model.pkl")

    # 用 PPO 继续训练
    ppo_trainer = PPOTrainer(model=model, env=env)
    ppo_trainer.train(num_episodes=1000)

    3. 混合训练（专家数据 + RL）:

    # 在 RL 训练中偶尔加入专家数据批次
    for episode in range(num_episodes):
        # RL 采集数据
        rl_data = collect_trajectories(env, model)

        # 偶尔添加专家数据
        if episode % 10 == 0:
            expert_batch = expert_dataset.get_batch(batch_size=32)
            rl_data.extend(expert_batch)

        # 更新模型
        model.update(rl_data)
    """)

    print("=" * 60)


if __name__ == "__main__":
    simple_behavior_cloning_demo()
    integration_with_training()

# RL Lessons So Far

Date: 2026-05-21

This note is the working rulebook for Orbit Wars RL experiments. Treat it as a
checklist before changing training, architecture, rewards, candidates, or
infrastructure.

## 中文整理

### 1. 第一优先级：环境必须快

RL 的第一件事是把环境改快。这不是可选项。如果环境慢到痛苦，就不要硬做 RL，会浪费时间。不要先幻想 sample efficiency；我们是在做 RL。

当前项目的本地路线是优先使用 fast simulator。只有在验证 Kaggle 兼容性、复现官方行为、或怀疑 simulator drift 时，才用 Kaggle 官方环境。

执行规则：

- 本地训练、数据生成、smoke check 默认使用 `training2.make_fast_orbit_wars`。
- 改 simulator 规则或碰撞逻辑后，必须跑 fast-vs-official compatibility check。
- 如果训练慢，先 profile 环境、MCTS、候选生成、模型推理各自耗时，再谈架构。

### 2. 架构搜索要看 signs of life

reward shaping 很重要。做架构搜索时，不是看 loss 好不好看，而是看策略是否会朝 reward structure 移动，这叫 signs of life。

执行规则：

- 每个架构变化都要回答：它有没有让 policy 朝我们想要的 reward/行为移动？
- 不要只看 train loss、value loss、entropy；必须看 rollout outcome、action/candidate 质量和策略行为。
- 如果 `avg_value` 长期全是 -1，哪怕 loss 稳定下降，也不算活了。

### 3. 一次只改一个 architecture delta

一次堆 7 个看起来都正确的变化，RL 里基本等于失去可解释性。每个变化单独看都合理，但训练坏了以后无法定位谁负责。

执行规则：

- 每次实验最多引入一个 architecture/training delta。
- 新变化必须和上一个可复现 baseline 对比。
- 实验命名要能看出唯一 delta。
- 如果同时改了多个东西，这轮只能当探索，不能当结论。

### 4. 工作 baseline 的“缺陷”可能在做正则化

一个看起来笨的 baseline 可能正因为笨才稳定。比如简单 FireHead、全局 head mixing、缺失某些 mask，可能不是 bug，而是在帮 PPO/训练过程压住梯度。

执行规则：

- 不要默认把 baseline 的奇怪限制当成 bug 修。
- 移除限制、增大容量、加细粒度 head 后，要同步检查学习率、warmup、entropy、clip/gradient 行为。
- 一个能训练的笨架构，比一个理论上聪明但训不动的架构更有价值。

### 5. Feature engineering 是工程问题

理想情况下 ML 会自己发现好表示，但我们没有无限规模。要主动把游戏机制、可观测结构、启发式先验塞进特征和候选空间里，让 policy search space 变小。

执行规则：

- 看 replay、看游戏、做分析，找出实际有用的状态和动作信息。
- 允许并鼓励把 game mechanics 变成 inductive bias。
- 不必先写完整 heuristic agent，但要理解 rulebase 为什么有效。
- 候选动作和特征比大模型更优先。

### 6. 2p reward 可以先用 +1 / -1

2p 模式下，先用简单胜负信号足够。不要太早把 reward 搞复杂。

执行规则：

- 先确认 +1/-1 下有 signs of life。
- 如果没有生命迹象，优先检查 value target、候选动作、MCTS 搜索、对手强度、player 分布，而不是立刻复杂 reward shaping。
- 注意 tie/排名/官方 reward 的语义必须一致。

### 7. Transformer RL 先用 canonical playbook

Transformer policy 训练痛苦是常态，不要先发明聪明技巧。IMPALA、AlphaStar、OpenAI Five 这类系统都强调 warmup、LR decay、entropy schedule 等基础训练纪律。

执行规则：

- transformer policy 默认考虑 warmup + cosine decay。
- entropy 不应只用一个粗糙常数；必要时考虑 per-head/per-action entropy。
- 看到 policy 变尖、entropy 掉、value head 变硬时，先减学习率或降容量。
- 不要在 vanilla settings 上硬扛两天再想起 warmup。

### 8. clip_frac 是早期报警器

在 PPO 语境里，clip_frac 单调爬升通常比 entropy collapse 或 KL spike 更早报警。它表示 optimizer 正输给 value/policy sharpening。

执行规则：

- 监控 clip_frac 从 0.10 往 0.30+ 的单调爬升。
- 出现爬升时，先降学习率、回退容量、调 entropy/warmup。
- 不要等到真正爆炸才处理。

备注：当前 `alphaZeroLike` 不是 PPO，但同样要保留“早期报警器”思维：例如 `policy_max` 单调上升、`entropy` 快速下降、`value_pred_mean` 过快贴到单一终局值，都要当警报。

### 9. AI assistant 的角色边界

AI 很适合写代码、port、parity tests、分析脚本、日志整理、机械重构。AI 不可靠的部分是：判断研究方向优先级、决定一条 run 是否死了、在训练结果上保持自洽、知道什么时候停。

执行规则：

- AI 建议要按 GPU 成本过滤；一天误导就是实打实的预算损失。
- 架构级决策、是否继续训练、是否提交，必须由人判断。
- AI 可以加速 tedious engineering，但不能替代实验判断。
- 每个 AI 提议都要问：验证它要花多少 GPU 时间？如果错了损失多少？

### 10. 回缩 baseline 的原则

当前目标不是堆功能，而是缩回一个能解释的 baseline。

执行规则：

- 从最小可解释链路开始：fast env -> candidate generator -> shallow search target -> policy/value update -> rollout eval。
- 一次只打开一个模块：proposal、MCTS depth、value target、reward shaping、multi-GPU。
- 如果训练稳定但 outcome 不变，先诊断 target 是否有信息，而不是改网络。
- 如果 proposal 预训练有效，RL 阶段默认先冻结 proposal，除非有证据证明联合训练有益。

### 11. Proposal 探索要先保护 draw

2026-05-21 的 `alphaZeroLike` 检查里，纯 rulebase anchor 在同 seed 的 20 局 180-step 对照为 `7W-7L-6D`，nonloss `65%`。直接加入模型 proposal 探索会把大量 draw 变成 loss：

- rulebase 附近候选扰动：`eps=0.02` 得到 `9W-11L-0D`，已经有探索信号，但 draw 稳定性下降。
- 模型 proposal-only 探索：`eps=0.05` 小样本 `1W-3L`；`eps=0.005` 在 20 局里 `7W-10L-3D`，只实际采用了 4 次 proposal。
- 提高 `proposal_send_threshold` 到 `0.55/0.65` 没有解决；说明问题不只是 source 数量，而是 proposal 的目标/时机质量。

执行规则：

- 不要把“rulebase 附近扰动能打”误读成“model proposal 能打”。
- proposal 探索必须记录 attempted / used / blocked，不然可能只是 fallback 在打。
- 第一阶段目标应拆开验收：先让 proposal imitation 的 source count、target top-k、action count 接近 rulebase；再放到在线探索。
- 在线探索必须有安全门：源数、总出兵量、必要时还要加 source remaining / target plausibility 过滤。
- 只要 draw 被系统性变成 loss，就先回到 proposal 质量诊断，不要启动 self-play 放大错误。

2026-05-21 后续结论：

- 直接训练 count loss 会把 `pred_action_size_mean` 从约 `5.3` 压到 `1.09`，但 source recall 掉到约 `0.315`，对局仍差，不作为 baseline。
- 小学习率 full-model target 微调没有突破 target top1，离线 target top1 仍约 `0.391`。
- 当前可解释 bridge：model proposal 只负责选择和 rulebase 重叠的 source，再把这些 source 投影回 rulebase 的目标/船量。40 局 180-step 同 seed 对照：
  - 纯 rulebase anchor：`16W-16L-8D`，nonloss `60%`，mean reward `0.2`。
  - proposal source projection，`eps=0.01/0.02`：`16W-19L-5D`，nonloss `52.5%`，mean reward `0.05`。
- 这说明 proposal 的 source signal 可以低频接入，但 target/ship head 还不能直接在线探索。

2026-05-22 后续结论：

- 更直接的 source-only bridge 优于 full proposal projection：只读取 proposal source 概率，在 rulebase action 内做 source 子集选择，target/ship 完全沿用 rulebase。
- 40 局 180-step：
  - `eps=0.02` source-prob bridge：`19W-17L-4D`，nonloss `57.5%`，mean reward `0.15`。
  - `eps=0.05` source-prob bridge：`22W-18L-0D`，win rate 高但 draw 归零，风险更大。
- 20 局 500-step：
  - 纯 rulebase anchor：`8W-8L-4D`，nonloss `60%`，mean reward `0.2`。
  - `eps=0.02` source-prob bridge：`9W-10L-1D`，nonloss `50%`，mean reward `0.0`。
- 当前第一阶段 baseline 候选是 source-prob bridge，下一步不要加探索率；要学“何时删 rulebase source 才不破坏 draw”。

2026-05-22 继续推进：

- Source-prob bridge 的阈值 sweep 显示，长局里 `proposal_send_threshold=0.30` 比 `0.40` 更稳：
  - `eps=0.02, threshold=0.30, 20x500`：`9W-9L-2D`，nonloss `55%`，mean reward `0.1`。
  - `eps=0.02, threshold=0.40, 20x500`：`9W-10L-1D`，nonloss `50%`，mean reward `0.0`。
- 平均每次有效探索约从 `4.43` 个 rulebase source 保留 `3.00` 个，删除 `1.43` 个。
- 加 `max_anchor_source_drops=1` 没有改善：`8W-10L-2D`，说明“机械限制最多删一个”不如降低阈值自然保守。
- 当前第一阶段 baseline 更新为：
  - `--proposal-explore-prob 0.02`
  - `--proposal-send-threshold 0.30`
  - `--proposal-explore-project-anchor-source-probs`
- 下一步不应再扫阈值；应诊断被删除 source 的语义，区分“冗余支援源”和“关键防守/补刀源”。

## English Source Summary

- Make the environment fast first. If the environment is painfully slow, do not
  attempt RL yet.
- Look for signs of life: the architecture must move its policy toward the
  reward structure.
- Treat RL as engineering. Add inductive bias through features, candidates, and
  game-mechanic understanding because we do not have unlimited scale.
- For 2p, +1/-1 reward is enough as a starting point.
- Add one architecture delta at a time. Stacking many changes destroys
  debuggability.
- A working baseline's limitations may be doing hidden regularization.
- Do not trust AI diagnosis to be self-consistent. Use AI for code, tests,
  analysis scripts, journals, and mechanical refactors; keep budget decisions
  with the human.
- clip_frac is an early warning sign in PPO-style training.
- Use the canonical transformer-RL playbook before clever tricks: warmup,
  cosine decay, entropy schedules, careful optimizer settings.
- AI cost is not only subscription cost; bad suggestions burn GPU time.

## Experiment Checklist

Before starting a new RL run:

- What is the single delta versus the last baseline?
- Is the environment fast path being used?
- Are player seats balanced?
- Is the value/reward target semantically correct?
- Is candidate generation producing legal and useful actions?
- Is there a cheap signs-of-life metric for this run?
- What exact metric makes us stop early?
- How much GPU time can this hypothesis spend?

After a run:

- Did rollout outcome improve, not just loss?
- Did policy entropy or policy max show premature sharpening?
- Did value predictions collapse to one constant?
- Did proposal quality degrade?
- Can the result be attributed to exactly one change?
- Should this become the new baseline, or should we revert?

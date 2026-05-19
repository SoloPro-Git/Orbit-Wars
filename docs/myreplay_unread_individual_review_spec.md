# myreplay/unread independent replay review and regular-rulebase spec

分析对象：`data/myreplay/unread/*.json`

前提：这些局里的 Solo 使用当前 regular rulebase：
`REGULAR_CONFIG = TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_P2TRICKLE_S30_P4MIDBORDER_S40_REGULAR_CONFIG`，
提交别名为 `tail_m2_max14_net7_overpay4_p4lowhome_active4_p2trickle_s30_p4midborder_s40_regular`。

这套 regular 已经不是基础 public rule，它已经启用了：

- early neutral bias，但只偏向 `production >= 3`、早期安全目标，且动态中立上限约为 18 ships。
- 4p opening territory penalty，`opening_territory_step_limit=35`，会惩罚更靠近敌人的中立星。
- capture hold margin gate，4p 高产目标需要额外 hold margin。
- recent loss recapture bias，但核心偏向 `production >= 4`、窗口 50、bonus 35。
- source threat send filter，4p 只在 `t=35..80` 对 `production >= 4`、半径 42 的源星做送船风险过滤。
- multiplayer diplomacy score，本地敌人 bonus 降到 3，neutral bonus 提到 5，leader prod bonus 只有 1。
- third-party tail capture/watchlist，用小船抢第三方刚打下来的尾刀，但单次最多 14 船且需要 post-capture overpay 4。
- 4p `active_players=4` early-neutral override：`t<=50` 时允许 `production >= 2` 的安全中立目标获得更高加分。
- 2p `active_players=2` opening high-production trickle：`t<=30` 时允许 `production >= 1`
  的源星用至少 5 船小额抢 `production >= 4` 且目标守军不超过 12 的中立星。
- 4p midgame border source reserve：`t=40..150`，只在 `active_players=4` 时对
  `production >= 2`、敌人半径 60 内的边境源星保留约 1 turn 产能和 margin 5，减少中盘扩张/进攻后被贴脸反打。

因此下面每局不再只说“扩张慢/防守弱”，而是判断：当前 regular 的这些规则在那一局为什么失效。

## 76932660 vs restricteur, 2p

胜者优秀策略：

restricteur 的开局不是单纯更早出手，Solo 在 `t=18` 已经先拿 planet 27。但 restricteur 选择的中立目标质量更好：到 `t=50` 已经是 3 星 / 7 产，Solo 是 3 星 / 5 产；到 `t=65` 直接拉到 7 星 / 12 产。它没有停在“拿几个便宜星”的阶段，而是持续把新产能转成下一轮扩张和进攻。`t=65` 开始拿 Solo/中立边缘星，随后 `t=82` 反吃 Solo 早期 planet 27，`t=103` 吃 Solo 的 production 3 planet 6，形成连续滚雪球。

Solo 失败原因：

这是 2p，所以 regular 的 4p territory、source threat send filter、home anchor 都不生效。剩下主要靠 arrival production、holdability、recent loss recapture。问题是 Solo 的目标选择太保守或太分散：`t=20` Solo 2 产但只剩 7 船，说明早期投入后没有形成足够产能；`t=65` 虽然 3 星 / 6 产，但 restricteur 已经 7 星 / 12 产。后续丢的很多是 production 1/2/3 星，recent-loss recapture 偏向 `prod>=4`，所以不会强力拉回。这里 regular 的短板是 2p 中低产星群的“数量优势”估值不足。

Spec:

- 2p 低产图增加 planet-count pressure：当敌方星数领先 3+，prod 1/2 的边境星也触发 recapture/denial。
- recent-loss recapture 在 2p 不应硬偏 `prod>=4`；可用 `enemy_cluster_count_gain` 补偿低产星价值。

## 76933032 vs Felipe Ferreira, 4p

胜者优秀策略：

Felipe 不是最早击穿 Solo 的玩家，早期主要是 schanitater 先打崩 Solo。Felipe 的优秀点是稳定扩张并避免卷入亏损乱战：`t=50` 5 星 / 9 产，`t=80` 4 星 / 12 产，`t=120` 7 星 / 15 产。它在别人互相消耗时保持可增长资产，后面从 `t=160` 的 12 星 / 25 产滚到 `t=200` 的 18 星 / 37 产。

Solo 失败原因：

Solo 开局其实不差，`t=35` 是 3 星 / 7 产。但 `t=48..74` 连续丢 planet 8、4、1、12、16，其中 planet 1 在 `t=67` 被 schanitater 拿走时剩 52 船，这是典型“已经被打穿但没有及时撤退/反制”的局。regular 的 4p territory penalty 只管 `t<=35` 的开局目标，之后没有识别“我正在被一侧玩家专门攻击”。multiplayer diplomacy 的 local enemy bonus 只有 3，leader bonus 只有 1，导致策略仍然可能继续按普通目标分数行动，而不是进入 anti-pinch survival。

Spec:

- 4p 增加 `pinch_alarm`：15 回合内丢 2 星或同一敌人连续打 Solo 星时，停止远端扩张，优先保 home cluster。
- 对非最终胜者造成的早期伤害也要响应；不要只盯 leaderboard leader。

## 76934377 vs Alvaro, 4p

胜者优秀策略：

Alvaro 开局非常会拿低守军高产中立：`t=8` 拿 prod4，`t=15` 拿 prod3，`t=29/30/35` 又连续拿 prod4/1/4/1。到 `t=35` 已经 9 星 / 24 产，而 Solo 6 星 / 18 产。Alvaro 中期继续拿 owned planets，`t=48..91` 连续吃别人星，`t=120` 到 17 星 / 44 产。

Solo 失败原因：

Solo 的早期 expansion 并不差，`t=20` 3 星 / 9 产，`t=35` 6 星 / 18 产。但 Solo 的高价值边境星太容易被回收：`t=41` 丢 prod4 planet 14，`t=51` 丢 prod3 planet 26，`t=91` 丢 prod4 planet 13，`t=103` 丢 planet 23 时对方剩 51。regular 有 capture hold gate，但它主要在发起捕获时估算能否 hold；实际局面里多方攻击和后续生产变化让这个 margin 不够。source-threat filter 只在 `t=35..80` 且生产 >=4，对 `t=91/103` 之后的关键丢星没有覆盖。

Spec:

- capture-hold 不只用于发射前，也要在 capture 后 30 回合持续计算守住概率。
- 4p source-threat filter 的结束时间从 80 扩到至少 130，且对 prod3 星也生效。

## 76934450 vs Oleh Patsan, 4p

胜者优秀策略：

Oleh 是强攻击型：`t=7` 就拿 prod3 中立，`t=18` 再拿 prod3，`t=20` 已 4 星 / 12 产。关键是它从 `t=27` 直接吃 Solo planet 0，`t=35` 吃 Solo planet 12，`t=45` 吃 Solo planet 4，边扩张边斩杀。到 `t=50` Oleh 已 8 星 / 25 产，Solo 只剩 2 星 / 7 产。

Solo 失败原因：

Solo 开局发射 11 次后基本失去行动能力，总船量 322，对方 6900。regular 的 `min_ships_mine_attack=12` 对这种母星初始高产但早期局势很凶的 4p 局可能过高，导致早期可行动窗口少。Solo 的 early-neutral 只拿了 4 个中立，并且 `t=27` 起连续丢星，没有进入“被 rush 防守”。recent-loss recapture 偏 prod4+，但 Solo 丢的 early home/低中产星同样决定生死。

Spec:

- 4p rush detection：`t<45` home 或近 home 星被攻击时，允许低于 12 船的防守/反打。
- early survival 比 ROI 更重要：被 rush 时临时关闭 distant neutral preference。

## 76934632 vs Caiden Matthews, 2p

胜者优秀策略：

Caiden 的优秀点是 2p 高产图上没有给 Solo 呼吸空间。`t=35` Caiden 已 10 星 / 32 产，Solo 6 星 / 24 产；`t=50` 16 星 / 47 产 vs Solo 12 星 / 37 产。随后 `t=54..80` 连续吃 Solo 的 prod3/prod5/prod4 星，把产能差从 10 扩到 41。

Solo 失败原因：

Solo 并不是开局完全失败，`t=50` 有 37 产。但当前 regular 在 2p 里没有足够强的“enemy production lead + wave attack”响应。Caiden `t=54` 吃 planet17，`t=58` 吃 prod5 planet12，`t=59` 吃 prod4 planet1，都是高价值星；regular 的 value defense 和 recapture 有，但被对方高 launch cadence 压住。`target_candidate_limit=2` 也可能导致每个源星只试前两个分数目标，错过最紧急的防守/反打目标。

Spec:

- 2p 若敌方 `prod_lead>=8` 且近 20 回合我方丢高产星，攻击候选从 2 提到 4。
- 对 prod4/5 星丢失，recapture bonus 应压过普通 neutral expansion。

## 76934678 vs Anirudh K, 2p

胜者优秀策略：

Anirudh 是持久高频压制。前 80 回合发射 124 次 / 3786 船，Solo 61 次 / 1298 船。它不一定每个时间点产能都碾压，`t=120..200` 两边产能接近，但 Anirudh 一直用更多舰队制造交换，最终在 `t=250` 后把 Solo 的星数和产能压下去。

Solo 失败原因：

这局最能说明 regular 的中后期主动性不足。Solo 到 `t=160` 仍然 16 星 / 38 产，不是输在开局；输在长期边境管理。Solo 从 `t=76` 起连续丢 planet29/3/19/7/31/15，很多是 prod1/2/4/5 混合。regular 的 recent-loss recapture 窗口和 bonus 会尝试拉回 prod4+，但低产星作为前线跳板不断丢失，导致战线退缩。规则缺少“前线形状/基地数量”价值。

Spec:

- 增加 front-base value：低产但位于前线、可作为 launch base 的星，防守价值上调。
- 中后期如果双方产能接近，tempo floor 要求每 20 回合至少有一定攻击/recapture 行动。

## 76935872 vs currypurin, 4p

胜者优秀策略：

currypurin 是典型 4p 快速地图控制。`t=20` 已 5 星 / 11 产，Solo 3 星 / 8 产；`t=50` 12 星 / 29 产，Solo 3 星 / 8 产。它早期既拿中立，也吃第三方/玩家星：`t=20` 就吃 Solo 刚相关的 planet30，`t=30` 又吃 Solo planet18。

Solo 失败原因：

Solo 的 `opening_territory_penalty` 可能在这类 4p 图中过度保守：它拿了几个近身目标，但没有和 currypurin 争夺中部/高价值扩张点。更严重的是 Solo 在 `t=20/t=30/t=34` 已连续丢星，但 regular 的 anti-pinch 没有触发；第三方 tail 逻辑是去抢别人尾刀，不是保护自己不被尾刀。到 `t=35` Solo 只剩 1 星 / 2 产，已经不可逆。

Spec:

- opening territory penalty 要加入“敌人已经在抢我方向资源”的反向逻辑，不能一味避开 contested。
- 4p 早期连续丢星时，启用 emergency recapture，不受 prod>=4 限制。

## 76936394 vs Thomas, 4p

胜者优秀策略：

Thomas 的核心不是最快开局，而是持续从别人的战斗中拿 owned planets。`t=50` Thomas 7 星 / 27 产，Solo 7 星 / 22 产，差距不大；但 Thomas 从 `t=30` 开始连续吃 owned planets，到 `t=100` 16 星 / 47 产，而 Solo 只剩 3 星 / 10 产。

Solo 失败原因：

Solo 的开局很可以：`t=50` 7 星 / 22 产。但从 `t=32` 丢 planet21，`t=45` 又丢同一星，`t=46` 丢 prod3 planet15，`t=68` 丢 prod3 planet20。这说明 regular 的 hold gate 和 source reserve 没有阻止“反复争夺同一批边境星”。home_anchor 只在 4p `t=60..130` 对 home radius 内源星留船，不能解决 `t=32..56` 的早期边境反复翻转。

Spec:

- 对重复翻转星建立 contested memory：同一 planet 30 回合内翻转 2 次，后续只有高胜率才继续投入。
- home_anchor 提前到 `t=35` 或为 early pinch 单独开 reserve。

## 76936647 vs automatylicza, 4p

胜者优秀策略：

automatylicza 的强点是极高发射频率和持续施压：前 80 回合 128 次 / 3484 船，Solo 24 次 / 502 船。`t=50` 双方同为 5 星 / 14 产，但之后 automatylicza 继续发射并进攻，`t=120` 到 10 星 / 28 产，Solo 只剩 1 星 / 4 产。

Solo 失败原因：

这局最像 regular 的 action throughput 问题。开局镜像阶段 Solo 还能跟上，`t=35/50` 都是 5 星 / 14 产。但 Solo 总发射只有 29 次，说明后续大量源星没有被充分利用，或被 reserve/安全过滤卡住。source-threat send filter 在 4p `t=35..80` 生效，可能过度拦截高产源星出船；同时 target_candidate_limit=2 让可行目标少时容易无动作。

Spec:

- 添加 “no-action fallback”：若本回合没有 moves，降低 reserve/候选限制，至少做安全短线 reinforcement/attack。
- 记录 source-threat filter 拒绝的 action，定位是否过度保守。

## 76937624 vs c-number, 4p

胜者优秀策略：

c-number 早期不领先，`t=35` Solo 5 星 / 15 产，c-number 4 星 / 11 产。但 c-number 很会把 Solo 的边境星打掉：`t=42` 吃 Solo prod2 planet15，`t=53` 吃 Solo prod3 planet6，`t=55` 吃 Solo prod3 planet19，`t=71` 吃 Solo prod3 planet17。到 `t=80` c-number 10 星 / 26 产，Solo 4 星 / 12 产。

Solo 失败原因：

这局说明 Solo 的早期扩张方向可以，但 holdability 低估了邻近敌人的反扑。regular 的 opening hold margin 对 production>=3 有帮助，但 `capture_hold_margin=4` 对 4p 夹击可能太低。且 Solo 丢的 planet15 是 prod2，不会被高优先级保护，但它可能是连接 cluster 的关键节点。

Spec:

- 4p hold margin 对“敌人两侧可达”的星提高到 8+。
- 防守评分加入 cluster connectivity，不只看 production。

## 76937889 vs SonOfNike, 2p

胜者优秀策略：

SonOfNike 早期专门吃大量低守军 prod5 星：`t=19/24/26/29` 连续拿 prod5/5/5/5。Solo 也拿了 prod5，但常常付出更多或来得更晚。到 `t=50` SonOfNike 16 星 / 36 产，Solo 10 星 / 30 产；`t=120` 已 18 星 / 46 产 vs Solo 10 星 / 22。

Solo 失败原因：

Solo 前 80 回合发射次数不低，甚至 59 次 vs 对方 48 次，所以不是简单 under-launch。失败在目标选择和防守：`t=71` 丢 planet15 时对方剩 46 船，`t=89` 丢 planet14 时对方剩 71 船，说明对手的 attack wave 不是小偷袭，是大规模主攻，Solo 没有提前识别。regular 的 enemy launch punish 只给目标评分加 bonus，不等价于“我方高产星即将被重兵打穿”的防守模式。

Spec:

- 增加 enemy wave detector：敌方同方向/同 cluster 大舰队出现时，切换到 preserve-prod mode。
- 对大船进攻目标，允许多源 reinforcement，即使 ROI 低也要保关键星。

## 76938253 vs HassenHamdi, 2p

胜者优秀策略：

HassenHamdi 的重点是超高频多点进攻：前 80 回合 168 次 / 2754 船，Solo 62 次 / 1372 船。它 `t=47..86` 几乎每几回合吃 Solo 一个星，且包括高剩余兵力的 planet20/12/15。

Solo 失败原因：

Solo 扩张还不错，`t=65` 13 星 / 29 产。但边境防线被完全穿透：`t=62` planet20 被吃后对方剩 64，`t=75` planet12 又被吃后剩 66，`t=81` planet15 剩 53，`t=86` planet15 剩 67。这不是 recapture bias 能解决的，因为对方留下的驻军太厚，说明防守发生得太晚。regular 需要 arrival-based reinforcement 更激进，而不是事后 recent-loss recapture。

Spec:

- 在 2p 中，value defense 应优先于进攻目标循环，并允许从多个 nearby sources 提前补。
- 对“对方到达后剩余 > 30”的预测，强制防守或撤退反打源头。

## 76938619 vs HilalElusive, 2p

胜者优秀策略：

HilalElusive 是极限微操/高频发射型：总发射 1072 次 / 38653 船，Solo 199 次 / 4678 船。双方 `t=80` 和 `t=120` 产能一度相等，但 Hilal 用舰队数量、连续小规模翻转和前线站位赢下长线。

Solo 失败原因：

这是 regular 最危险的对手类型：规则每源星最多试少量候选，且每回合动作较少；面对大量小波次时，reactive defense 会疲于奔命。Solo 从 `t=66` 开始丢一串 prod1/2/3 星，看起来单个都不值得重兵防，但合起来让 Hilal 获得 launch base 和包围角度。recent-loss recapture 偏高产，低产前线被忽略。

Spec:

- 增加 low-prod front attrition detector：20 回合内低产星净损失 >=3 时，前线低产星权重翻倍。
- 提高 custom candidate breadth 或增加批量 low-risk harassment。

## 76938983 vs Alexey Yurasov, 2p

胜者优秀策略：

Alexey 在低产慢图上理解了“星数就是产能/发射点”。`t=80` Alexey 11 星 / 13 产，Solo 4 星 / 6 产；虽然单星产能不高，但 Alexey 用更多基地持续扩散，到 `t=250` 24 星 / 42 产。

Solo 失败原因：

regular 的 scoring 偏经济 ROI 和 production，低产图上容易低估 prod1/2 星的空间价值。Solo 前 80 回合发射次数与对方接近，说明不是懒，而是选择不对：拿了少量星后没建立足够发射网络。recent-loss recapture 也几乎不会强救 prod1 星，导致 Alexey 反复吃低产前线。

Spec:

- 低产图模式：当全图平均 production 低时，planet count 权重大幅提高。
- prod1/2 但距离近、可作跳板的星应进入目标候选前列。

## 76939353 vs Fishman97, 4p

胜者优秀策略：

Fishman97 开局中立不多，`t=50` 只有 2 星 / 8 产，但它打 owned planets 很有效：`t=58` 同时吃玩家 2 和 Solo 的星，`t=60` 吃 Solo prod5 planet17。它不是靠早期 neutral flood 赢，而是靠在混战中挑可盈利目标，最后长期存活并收割。

Solo 失败原因：

Solo `t=35/50` 有 5 星 / 11 产，领先 Fishman97，但 `t=58..60` 连续丢 planet13/12/17，尤其 prod5 planet17。regular 的 4p diplomacy 可能把 Fishman97 当作非本地/非leader威胁处理，leader bonus 又只有 1，没能识别“虽然它星少，但它正在高质量吃人”。third-party tail 逻辑偏抢尾刀，不足以防止自己成为别人的高 ROI owned target。

Spec:

- 增加 threat-by-recent-captures：短期 owned capture 质量高的敌人，即使不是 leader，也提高危险权重。
- 4p 防守不应只看 leader，应该看 who is attacking me profitably。

## 76939621 vs Sitoa, 4p

胜者优秀策略：

Sitoa 前期敢拿大目标：`t=14` 拿 prod5 planet0，剩 21 船；`t=21/27/34` 又拿一批 prod2 星。中期它没有被 Solo `t=65` 的短暂产能领先吓住，而是在 `t=80` 后持续吃 Solo/别人星，`t=120` 已 16 星 / 47 产。

Solo 失败原因：

Solo `t=65` 有 7 星 / 22 产，甚至高于 Sitoa 的 6 星 / 18，但随后开始崩：`t=80` 丢 prod4 planet4，`t=108` 丢 prod3 planet10 时对方剩 62。regular 的 source-threat send filter 只到 `t=80`，刚好在后续关键阶段失效；recent loss recapture 有窗口，但对方留下的守军太多，事后抢不回来。

Spec:

- 4p source-threat filter/defense window 扩展到 `t=120`。
- 对敌方已在高产星留下大驻军的情况，策略应转为攻击其薄弱生产线，而不是继续尝试正面 recapture。

## 76939868 vs Blu3s, 2p

胜者优秀策略：

Blu3s 是高质量扩张 + 中期集中攻势。`t=50` 12 星 / 42 产，Solo 10 星 / 37 产，差距还不大；但 `t=56..71` 连续吃 Solo 7 个星，其中 planet15 是 prod5 且对方剩 73。它把很小的早期优势转成一波不可逆 attack wave。

Solo 失败原因：

Solo 前 80 回合发射 74 次 / 2087 船，不算低。失败点是没有读出 Blu3s 的主攻节奏。regular 的 enemy launch punish 是“对敌方刚发射后可能空虚的星加分”，但如果敌方发射的是压死我高产星的大波，punish 源头可能来不及，应该先保目标。Solo 丢 prod5/prod4/prod3 连续发生，说明 value defense 不够硬。

Spec:

- enemy launch punish 与 value defense 冲突时，若我方目标星 prod>=4 且预计失守，value defense 优先。
- attack wave 后，优先反击对方低防高产源，而非普通 neutral。

## 76940240 vs galaxy2025, 4p

胜者优秀策略：

galaxy2025 是超高频 4p 控图，前 80 回合 205 次 / 5671 船，总发射 1509 次 / 59843 船。它 `t=35` 已 4 星 / 15 产，`t=50` 7 星 / 22 产，之后通过大量小动作不断吃边境和第三方资源。Solo 到 `t=200` 仍能接近，但后期被 action volume 和前线基地数压垮。

Solo 失败原因：

Solo 在这局不是早早死掉，`t=120..200` 产能还接近，说明 regular 的基本扩张可用。但面对 galaxy2025 这种高频对手，Solo 的 target_candidate_limit、单源行动、保守过滤导致动作密度完全不足。4p third-party tail 虽然存在，但最多 14 船且偏机会主义，无法替代主动控图。

Spec:

- 对高频敌人增加 `pressure_response`: 我方每 20 回合动作数低于 leader 50% 时，扩大候选并降低 min-send。
- 增加 safe harassment/denial，小船扰乱敌人低守军扩张。

## 76940470 vs Felipe Ferreira, 4p

胜者优秀策略：

这是低产慢图。Felipe 没有很早碾压，但长期保持更好的星数和翻转效率。`t=160` 两边都是 10 星 / 14 产，但 `t=300` Felipe 13 星 / 19 产，Solo 7 星 / 9 产。它在低产图上把每一个 base 都留住并逐步转换成优势。

Solo 失败原因：

Solo 在 `t=69..122` 丢了大量 prod1/3 星：planet4、16、0、14、12、17 等。regular 在低产图上仍然把很多 prod1 星看得太轻，recapture 偏 prod4+ 基本帮不上忙。低产局里一个 prod1 星也可能是发射位置和地图控制点，不能只按经济产能算。

Spec:

- low-production map mode：全图高产少时，所有 owned planet 的防守基础值提高。
- recapture min production 根据地图平均产能动态降低。

## 76940742 vs higaki, 4p

胜者优秀策略：

higaki 不是高频开局，前 80 回合只 13 次 / 532 船，少于 Solo 的 30 次 / 614 船。但它的 owned-planet 攻击质量高：`t=38` 吃玩家0 prod3，`t=40` 吃 Solo 方附近玩家3星，`t=75` 吃 Solo prod2 planet6，后面逐步收割。它赢在选择攻击对象和后续保有，而不是单纯动作多。

Solo 失败原因：

Solo `t=80` 与 higaki 同为 21 产，甚至早期星数也不错。但 Solo 的边境星不断被不同玩家吃：`t=36` 丢 prod3 planet22，`t=45` 丢 planet26，`t=68` 丢 prod3 planet14，`t=75` 丢 planet6，`t=84` 丢 prod3 planet13。regular 的 4p diplomacy local bonus 太弱，无法处理“多邻居轮流咬我”的局面；它也缺少 cluster-level fortification。

Spec:

- 4p 多敌攻击计数：不同敌人在 30 回合内各吃 Solo 星时，触发 multi-front defense。
- 选择保一个 compact cluster，而不是平均扩张和平均反击。

## Cross-replay implementation plan

### Plan A: replay-aware diagnostics

新增 replay audit，不只输出统计，还要输出规则失败原因：

- 每个 Solo 发射被哪些 gate/score 影响：target_candidate_limit、source threat、hold gate、territory penalty、recent recapture。
- 每个 Solo 丢星前 20 回合是否有可用 reinforcement source。
- 每个丢星是否满足 current regular 的 recapture 条件；不满足时记录原因：prod 太低、窗口外、候选没进前 2、源星不安全。
- 对 4p，记录攻击 Solo 的敌人是否为 leader、local enemy、recent high-quality capturer。

### Plan B: regular counter-spec

优先实现这些最小改动：

1. 2p/低产图 front-base value：低产前线星也要守/抢。
2. enemy wave detector：大波攻击我方高产星时，防守优先于 enemy-launch punish。
3. 4p pinch/multi-front alarm：连续丢星或多敌攻击时进入 survival mode。
4. recapture dynamic min production：低产图、2p 星数落后、front-base 丢失时，允许 prod1/2/3 recapture。
5. candidate breadth fallback：无动作或连续低动作时，从 top2 扩到 top4/top6。
6. source-threat/hold defense window：4p 从 `t=80` 扩到 `t=120/140`，并覆盖 prod3。

## Ablation notes after current regular

已完成的 fast simulator / numba 消融结果：

- `myreplay_plan021_p2_low_home_trickle_confirm`，24 games/variant，`workers=2`：
  `p2_low_home_trickle_s30_src1_tgt4_m5` 为 15-9，regular 为 7-17；
  `s40_src1_tgt3_m5` 为 10-14。结论：2p 低产母星开局提前向 prod4 中立小额分流是强正信号。
- `myreplay_plan022_p2_trickle_highprod_combo`，40 games/variant，`workers=64`：
  单独 `p2_trickle_s30` 为 24-16，regular 为 13-26-1；
  叠加 `send_filter_s45_115_m6` 降到 21-19，叠加 `recent_reserve_s45_150_m6` 降到 20-20。结论：主收益来自 opening trickle，不来自中盘源星防守组合。
- `myreplay_plan015_p2_highprod_source_guard_focus`，40 games/variant，`workers=64`：
  `send_filter_s45_115_m6` 和 `recent_reserve_s45_150_m6` 都是 18-21-1，regular 是 13-26-1。结论：高产防守单独有弱正信号，但不应压过 trickle。
- `myreplay_plan020_leader_pressure_focus`，40 games/variant，`workers=64`：
  `leader_pressure_125` 为 13-27，regular 为 12-28；`leader_pressure_175` 反而 11-29。结论：4p leader pressure 不够稳，暂不升。
- `myreplay_plan014_early_4p_source_guard`，40 games/variant，`workers=64`：
  `p4_early_send_filter_s35_90_m6` 为 13-27，regular 为 12-28。结论：只有微正，暂不升。
- `myreplay_plan019_p4_low_home_opening_confirm`，80 games/variant，`workers=64`：
  旧 `tail_m2_max14_net7_overpay4_regular` 为 19-61，新 `regular` / `p4_low_home_bonus_prod2_s50_active4` 为 17-63。结论：新 4p low-home override 在这组池里没有继续领先；仍需依赖之前 2P/历史高分池“不伤”的验证记录。
- `myreplay_plan023_p2_frontbase_recap_breadth`，40 games/variant，`workers=64`：
  新 regular 为 25-15，`p2_recap_prod2_b30` / `p2_recap_prod1_b24` 都为 23-17，
  `p2_candidate_limit3` 为 20-20，`p2_front_support_soft` 为 3-37。结论：不要把前线支援/候选扩宽收进 regular。
- `myreplay_plan024_p4_pinch_front_defense`，48 games/variant，`workers=64`：
  `p4_proactive_recent_p3_s55_150` 与 regular 同为 13-35，
  local reserve/front support 组合明显负向。结论：前线支援和本地 reserve 太容易锁死进攻节奏。
- `myreplay_plan026_p4_source_gate_border`，48 games/variant，`workers=64`：
  `p4_mid_border_prod2_s45_150` 为 16-32，regular 为 13-35。结论：4p 中盘边境源星保留有小正信号。
- `myreplay_plan027_p4_mid_border_confirm`，80 games/variant，`workers=64`：
  `p4_mid_border_prod2_s40_150_m5` 为 20-60，regular 为 17-63。结论：提前到 step 40、margin 5 的版本最好。
- `myreplay_plan028_p4_mid_border_top_confirm`，120 games/variant，`workers=64`：
  `p4_mid_border_prod2_s40_150_m5` 为 36-84，regular 为 30-90。结论：三轮同向小正，且该 gate 只在 4p 生效，收进 regular。
- `myreplay_plan029_p2_no_attack_fallback`，80 games/variant，`workers=64`：
  regular 为 43-37，`p2_no_attack_fallback_c5` 为 26-54，
  `c6_min11` 为 15-65，`c8_min10` 为 12-68。结论：无攻击后补打一发会显著破坏 2p 节奏，开关保留但默认关闭，不收进 regular。

当前已晋级 regular 的配置：
`TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_P2TRICKLE_S30_P4MIDBORDER_S40_REGULAR_CONFIG`。
旧 regular `TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_P2TRICKLE_S30_REGULAR_CONFIG`
和 `TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_REGULAR_CONFIG` 保留为命名历史配置。

### Validation

先用 fast simulator 验证 regular 变体：

```bash
uv run python rulebase/kaggle_public_rl_informed_strategies/fast_ablation_eval.py --opponent regular --games 200 --players 2
uv run python rulebase/kaggle_public_rl_informed_strategies/fast_multiplayer_eval.py --games 200
```

如果改到 simulator/collision 逻辑，再按 AGENTS.md 跑官方兼容比较；纯策略改动不需要优先跑官方环境。

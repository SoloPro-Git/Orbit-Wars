# Orbit Wars - 游戏规则

## Overview

玩家从一个母星出发，通过向中立和敌方星球发送舰队来争夺地图控制权。棋盘为 100x100 连续空间，中心有一个太阳。星球围绕太阳公转，彗星沿椭圆轨道飞过，舰队沿直线飞行。游戏持续 500 回合。最终拥有最多飞船（星球上 + 舰队中）的玩家获胜。

## Board Layout（棋盘布局）

- **Board**: 100x100 连续空间，原点在左上角
- **Sun**: 位于 (50, 50)，半径 10。穿越太阳的舰队会被摧毁
- **Symmetry**: 所有星球和彗星按中心四重镜像对称放置：`(x, y), (100-x, y), (x, 100-y), (100-x, 100-y)`，确保公平性

## Planets（星球）

每个星球表示为 `[id, owner, x, y, radius, ships, production]`

- **owner**: 玩家 ID (0-3)，-1 表示中立
- **radius**: 由产量决定：`1 + ln(production)`，高产量星球物理上更大
- **production**: 1 到 5 的整数，每个回合拥有的星球生成该数量的飞船
- **ships**: 当前驻军，初始值在 5 到 99 之间（偏向低值）

### Planet Types（星球类型）

- **公转星球**: `orbital_radius + planet_radius < 50` 的星球以恒定角速度（0.025-0.05 弧度/回合，每局随机）绕太阳公转。使用 `initial_planets` 和 `angular_velocity` 预测位置
- **静止星球**: 距中心较远的星球不旋转

地图包含 20-40 个星球（5-10 个对称组，每组 4 个）。至少 3 组静止，至少 1 组公转。

### Home Planets（母星）

随机选择一个对称组作为起始星球。2 人游戏中，玩家在对角星球（Q1 和 Q4）出发。4 人游戏中，每个玩家获得组中的一个星球。母星初始有 10 艘飞船。

## Fleets（舰队）

每个舰队表示为 `[id, owner, x, y, angle, from_planet_id, ships]`

- **angle**: 移动方向（弧度）
- **ships**: 舰队中的飞船数量（航行中不变）

### Fleet Speed（舰队速度）

舰队速度按对数曲线缩放：

```
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
```

- 1 艘飞船速度为 1.0 单位/回合
- 更大的舰队更快，趋近最大速度（默认 6.0）
- ~500 艘飞船速度约 5，~1000 艘达到最大速度

### Fleet Movement（舰队移动）

舰队每回合沿直线移动。舰队在以下情况被移除：

- 越界（离开 100x100 棋盘）
- 穿过太阳（路径段进入太阳半径范围内）
- 与任何星球碰撞（路径段进入星球半径范围内），触发战斗

碰撞检测是连续的——检查从旧位置到新位置的整个路径段，不仅仅是终点。

### Fleet Launch（舰队发射）

每回合，agent 返回移动列表：`[from_planet_id, direction_angle, num_ships]`

- 只能从自己拥有的星球发射
- 不能发射超过星球当前拥有的飞船数
- 舰队在星球半径外侧沿指定方向生成
- 可以在同一回合从同一或不同星球发起多次发射

## Comets（彗星）

彗星是临时性太阳系外天体，沿围绕太阳的高椭圆轨道飞过棋盘。它们在第 50、150、250、350、450 回步以 4 个一组（每个象限一个）生成。

- **Radius**: 1.0（固定）
- **Production**: 拥有时每回合 1 艘飞船
- **Starting ships**: 随机，偏向低值（最少 4 次 1-99 的随机取值）。一组中的 4 颗彗星共享相同的初始飞船数
- **Speed**: 可通过 `cometSpeed` 配置（默认 4.0 单位/回合）
- **Identification**: 查看 observation 中的 `comet_planet_ids` 了解哪些星球 ID 是彗星。彗星也出现在 `planets` 数组中，遵循所有普通星球规则

当彗星离开棋盘时，连同上面的驻军一起被移除。彗星在每回合的舰队发射前被移除，因此不能从即将离开的彗星发射。

`comets` observation 字段包含彗星组数据，包括 `paths`（每颗彗星的完整轨迹）和 `path_index`（沿路径的当前位置），可用于预测彗星未来位置。

## Turn Order（回合顺序）

每回合按以下顺序执行：

1. **彗星过期**: 移除已离开棋盘的彗星
2. **彗星生成**: 在指定回合生成新彗星组
3. **舰队发射**: 处理所有玩家操作，创建新舰队
4. **生产**: 所有拥有的星球（包括彗星）生成飞船
5. **舰队移动**: 移动所有舰队，检查越界、太阳碰撞和星球碰撞。撞上星球的舰队排队等待战斗
6. **星球旋转 & 彗星移动**: 公转星球旋转，彗星沿路径前进。被移动星球/彗星扫到的舰队进入战斗
7. **战斗结算**: 解决所有排队的星球战斗

## Combat（战斗）

当一个或多个舰队与星球碰撞时（飞入或被移动的星球扫到），战斗结算如下：

1. 所有到达的舰队按所有者分组，同一所有者的飞船数相加
2. 最大攻击力量与第二大攻击力量交战，差值的飞船存活
3. 如果有存活的攻击者：
   - 如果攻击者与星球拥有者相同，存活飞船加入驻军
   - 如果攻击者是不同拥有者，存活飞船与驻军交战。如果攻击者超过驻军，星球易主，驻军变为剩余数
4. 如果两个攻击者平局，所有攻击飞船被摧毁（无存活）

## Scoring and Termination（计分与终止）

游戏在以下情况结束：

- **步数上限**: 500 回合
- **淘汰**: 只剩一个玩家（或零个）拥有任何星球或舰队

最终得分 = 拥有星球上的飞船总数 + 拥有舰队中的飞船总数。最高分获胜。

## Observation Reference（观测数据）

| Field | Type | Description |
|-------|------|-------------|
| `planets` | `[[id, owner, x, y, radius, ships, production], ...]` | 所有星球（包括彗星） |
| `fleets` | `[[id, owner, x, y, angle, from_planet_id, ships], ...]` | 所有活跃舰队 |
| `player` | `int` | 你的玩家 ID (0-3) |
| `angular_velocity` | `float` | 星球旋转速度（弧度/回合） |
| `initial_planets` | `[[id, owner, x, y, radius, ships, production], ...]` | 游戏开始时的星球位置 |
| `comets` | `[{planet_ids, paths, path_index}, ...]` | 活跃彗星组数据 |
| `comet_planet_ids` | `[int, ...]` | 彗星的星球 ID |
| `remainingOverageTime` | `float` | 剩余超时时间预算（秒） |

## Action Format（操作格式）

返回移动列表：`[[from_planet_id, direction_angle, num_ships], ...]`

- `from_planet_id`: 你拥有的星球 ID
- `direction_angle`: 弧度角度（0 = 右，pi/2 = 下）
- `num_ships`: 发送的飞船数量

返回空列表 `[]` 表示不采取行动。

## Agent Convenience（Agent 便捷工具）

模块导出命名元组以便于字段访问：

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

def agent(obs):
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    player = obs.get("player", 0)
    for p in planets:
        print(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
    return []  # list of [from_planet_id, angle, num_ships]
```

## Configuration（配置）

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodeSteps` | 500 | 最大回合数 |
| `actTimeout` | 1 | 每回合秒数 |
| `shipSpeed` | 6.0 | 最大舰队速度 |
| `sunRadius` | 10.0 | 太阳半径 |
| `boardSize` | 100.0 | 棋盘尺寸 |
| `cometSpeed` | 4.0 | 彗星速度（单位/回合） |

## Timeline（时间线）

- **2026年4月16日** - 开始日期
- **2026年6月16日** - 参赛截止日期（必须在此之前接受比赛规则）
- **2026年6月16日** - 团队合并截止日期
- **2026年6月23日** - 最终提交截止日期
- **2026年6月24日至约7月8日** - 继续运行比赛直到排行榜收敛

## Prizes（奖金）

- 第1名 - $5,000
- 第2名 - $5,000
- 第3名 - $5,000
- 第4名 - $5,000
- 第5名 - $5,000
- 第6名 - $5,000
- 第7名 - $5,000
- 第8名 - $5,000
- 第9名 - $5,000
- 第10名 - $5,000

## Evaluation（评估）

每天可提交最多 5 个 agent。每个提交会与技能评级相似的其他 bot 对战。系统只追踪最新的 2 个提交作为最终提交。

每个提交有一个估计的技能评级，由高斯分布 N(μ,σ²) 建模。上传时初始化为 μ₀=600。胜利增加 μ，失败减少 μ。胜负的分差不影响技能评级更新。

在 2026年6月23日提交截止后，额外提交将被锁定，继续运行约两周后排行榜最终确定。

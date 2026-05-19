from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.expert.action_labeling import infer_target_planet_id


PHASES = [
    ("opening_0_50", 0, 50),
    ("early_51_120", 51, 120),
    ("mid_121_220", 121, 220),
    ("late_221_end", 221, 10_000),
]


def fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    ratio = max(0.0, min(1.0, math.log(max(ships, 1.0)) / math.log(1000.0)))
    return 1.0 + 5.0 * (ratio**1.5)


def dist(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def score(obs: dict[str, Any], player: int) -> float:
    return float(
        sum(float(p[5]) for p in obs.get("planets", []) if int(p[1]) == player)
        + sum(float(f[6]) for f in obs.get("fleets", []) if int(f[1]) == player)
    )


def prod(obs: dict[str, Any], player: int) -> int:
    return int(sum(int(p[6]) for p in obs.get("planets", []) if int(p[1]) == player))


def planet_count(obs: dict[str, Any], player: int) -> int:
    return sum(1 for p in obs.get("planets", []) if int(p[1]) == player)


def by_id(planets: list[list[Any]]) -> dict[int, list[Any]]:
    return {int(p[0]): p for p in planets}


def phase_name(step: int) -> str:
    for name, lo, hi in PHASES:
        if lo <= step <= hi:
            return name
    return PHASES[-1][0]


def target_class(target: list[Any] | None, player: int) -> str:
    if target is None:
        return "unknown"
    owner = int(target[1])
    if owner == player:
        return "friendly"
    if owner == -1:
        return "neutral"
    return "enemy"


@dataclass
class ActionEvent:
    step: int
    source: int
    target: int | None
    target_owner: int | None
    target_prod: int | None
    target_ships: float | None
    ships: int
    source_ships_before: float
    source_ships_after: float
    sent_ratio: float
    eta: float | None


@dataclass
class FlipEvent:
    step: int
    planet: int
    prev_owner: int
    new_owner: int
    ships_after: float
    production: int


def load_frames(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    frames = []
    for step_row in data["steps"]:
        if not step_row:
            continue
        obs = step_row[0].get("observation")
        if obs:
            frames.append(obs)
    return data, frames


def capture_time(frames: list[dict[str, Any]], start_step: int, pid: int | None, player: int) -> int | None:
    if pid is None:
        return None
    for obs in frames:
        step = int(obs.get("step", 0))
        if step < start_step:
            continue
        p = by_id(obs.get("planets", [])).get(pid)
        if p is not None and int(p[1]) == player:
            return step
    return None


def summarize_game(path: Path) -> dict[str, Any]:
    data, frames = load_frames(path)
    teams = data.get("info", {}).get("TeamNames", [])
    names = [str(x) for x in teams]
    v_candidates = [i for i, n in enumerate(names) if n.lower() == "vadasz"]
    if not v_candidates:
        v_candidates = [i for i, n in enumerate(names) if "vadasz" in n.lower()]
    if not v_candidates:
        raise ValueError(f"Vadasz not found in {path}")

    rewards = data.get("rewards", [])
    v_player = max(v_candidates, key=lambda i: float(rewards[i]) if i < len(rewards) else -999.0)
    result = "win" if float(rewards[v_player]) > 0 else "loss" if float(rewards[v_player]) < 0 else "draw"
    opponents = [f"P{i}:{n}" for i, n in enumerate(names) if i != v_player]

    action_events: list[ActionEvent] = []
    phase_stats: dict[str, dict[str, Any]] = {
        name: {
            "launches": 0,
            "ships": 0,
            "classes": Counter(),
            "ratios": [],
            "multi_action_turns": 0,
            "turns_with_action": 0,
        }
        for name, _, _ in PHASES
    }
    action_turn_counts: Counter[int] = Counter()

    for step_row in data["steps"]:
        if v_player >= len(step_row):
            continue
        agent = step_row[v_player]
        obs = agent.get("observation") or {}
        step = int(obs.get("step", step_row[0].get("observation", {}).get("step", 0) or 0))
        actions = agent.get("action") or []
        if actions:
            action_turn_counts[step] += 1
            phase_stats[phase_name(step)]["turns_with_action"] += 1
            if len(actions) >= 2:
                phase_stats[phase_name(step)]["multi_action_turns"] += 1
        planets = by_id(obs.get("planets", []))
        outgoing_by_source: Counter[int] = Counter()
        for raw in actions:
            if len(raw) >= 3:
                outgoing_by_source[int(raw[0])] += int(raw[2])
        for raw in actions:
            if len(raw) < 3:
                continue
            source_id, angle, ships = int(raw[0]), float(raw[1]), int(raw[2])
            source = planets.get(source_id)
            target_id = infer_target_planet_id(obs, source_id, angle, ships)
            target = planets.get(target_id) if target_id is not None else None
            source_ships_after = float(source[5]) if source else 0.0
            source_ships_before = source_ships_after + float(outgoing_by_source[source_id])
            eta = None
            if source and target:
                eta = dist(source, target) / max(fleet_speed(ships), 1e-6)
            ev = ActionEvent(
                step=step,
                source=source_id,
                target=target_id,
                target_owner=int(target[1]) if target else None,
                target_prod=int(target[6]) if target else None,
                target_ships=float(target[5]) if target else None,
                ships=ships,
                source_ships_before=source_ships_before,
                source_ships_after=source_ships_after,
                sent_ratio=(ships / source_ships_before) if source_ships_before > 0 else 0.0,
                eta=eta,
            )
            action_events.append(ev)
            ps = phase_stats[phase_name(step)]
            ps["launches"] += 1
            ps["ships"] += ships
            ps["classes"][target_class(target, v_player)] += 1
            ps["ratios"].append(ev.sent_ratio)

    flips: list[FlipEvent] = []
    prev_planets = by_id(frames[0].get("planets", [])) if frames else {}
    for obs in frames[1:]:
        cur_planets = by_id(obs.get("planets", []))
        for pid, cur in cur_planets.items():
            prev = prev_planets.get(pid)
            if prev is None:
                continue
            if int(prev[1]) != int(cur[1]):
                flips.append(
                    FlipEvent(
                        step=int(obs.get("step", 0)),
                        planet=pid,
                        prev_owner=int(prev[1]),
                        new_owner=int(cur[1]),
                        ships_after=float(cur[5]),
                        production=int(cur[6]),
                    )
                )
        prev_planets = cur_planets

    checkpoints = []
    if frames:
        wanted = [0, 20, 40, 60, 80, 120, 160, 220, 320, int(frames[-1].get("step", 0))]
        seen = set()
        for w in wanted:
            obs = min(frames, key=lambda o: abs(int(o.get("step", 0)) - w))
            step = int(obs.get("step", 0))
            if step in seen:
                continue
            seen.add(step)
            players = sorted({i for i in range(len(names))} | {int(p[1]) for p in obs.get("planets", []) if int(p[1]) >= 0})
            scores = {p: score(obs, p) for p in players}
            prods = {p: prod(obs, p) for p in players}
            v_score = scores.get(v_player, 0.0)
            best_other = max((s for p, s in scores.items() if p != v_player), default=0.0)
            checkpoints.append(
                {
                    "step": step,
                    "v_planets": planet_count(obs, v_player),
                    "v_prod": prods.get(v_player, 0),
                    "v_score": round(v_score, 1),
                    "best_other_score": round(best_other, 1),
                    "score_delta": round(v_score - best_other, 1),
                    "prod_delta": prods.get(v_player, 0) - max((v for p, v in prods.items() if p != v_player), default=0),
                }
            )

    first_actions = []
    for ev in action_events[:18]:
        cap = capture_time(frames, ev.step, ev.target, v_player)
        first_actions.append({**ev.__dict__, "capture_step": cap})

    gained = [f for f in flips if f.new_owner == v_player]
    lost = [f for f in flips if f.prev_owner == v_player and f.new_owner != v_player]
    high_value_gains = [f for f in gained if f.production >= 3 or f.prev_owner >= 0][:16]
    painful_losses = [f for f in lost if f.production >= 2 or f.ships_after >= 20][:16]
    repeated = Counter(f.planet for f in flips)

    return {
        "file": path.name,
        "episode": data.get("info", {}).get("EpisodeId", path.stem),
        "teams": names,
        "v_player": v_player,
        "opponents": opponents,
        "players": len(names),
        "result": result,
        "reward": rewards[v_player] if v_player < len(rewards) else None,
        "length": int(frames[-1].get("step", 0)) if frames else 0,
        "seed": data.get("info", {}).get("seed"),
        "action_events": action_events,
        "first_actions": first_actions,
        "phase_stats": phase_stats,
        "flips": flips,
        "gained": gained,
        "lost": lost,
        "high_value_gains": high_value_gains,
        "painful_losses": painful_losses,
        "repeated_planets": repeated,
        "checkpoints": checkpoints,
    }


def fmt_owner(owner: int | None, v_player: int) -> str:
    if owner is None:
        return "未知"
    if owner == -1:
        return "中立"
    if owner == v_player:
        return "己方"
    return f"敌方P{owner}"


def phase_line(name: str, stats: dict[str, Any]) -> str:
    cls = stats["classes"]
    ratios = stats["ratios"]
    ratio = f"{mean(ratios):.2f}" if ratios else "0.00"
    return (
        f"- `{name}`: {stats['launches']} 次 / {stats['ships']} ships；"
        f"目标 中立{cls['neutral']} 敌方{cls['enemy']} 支援{cls['friendly']} 未命中{cls['unknown']}；"
        f"平均出兵比例 {ratio}；多动作回合 {stats['multi_action_turns']}。"
    )


def describe_game(g: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    result_cn = {"win": "胜", "loss": "负", "draw": "平"}[g["result"]]
    lines.append(
        f"### {g['episode']} `{g['file']}`：Vadasz P{g['v_player']} {result_cn}，"
        f"对手 {', '.join(g['opponents'])}，长度 t={g['length']}"
    )
    lines.append("")
    lines.append("**关键时间线**")
    lines.append("")
    for cp in g["checkpoints"]:
        lines.append(
            f"- t={cp['step']}: 星球 {cp['v_planets']}，产能 {cp['v_prod']}，"
            f"总船 {cp['v_score']}，相对最强对手船差 {cp['score_delta']}，产能差 {cp['prod_delta']}。"
        )
    lines.append("")
    lines.append("**开局逐手**")
    lines.append("")
    for i, ev in enumerate(g["first_actions"], start=1):
        cap = ev["capture_step"]
        cap_s = f"t={cap} 到手" if cap is not None else "未在后续稳定到手"
        eta_s = f"{ev['eta']:.1f}" if ev["eta"] is not None else "?"
        lines.append(
            f"- #{i} t={ev['step']}: P{ev['source']} -> "
            f"{'?' if ev['target'] is None else 'P' + str(ev['target'])}，"
            f"发 {ev['ships']}/{ev['source_ships_before']:.0f} ({ev['sent_ratio']:.0%})，"
            f"发后余 {ev['source_ships_after']:.0f}，"
            f"目标{fmt_owner(ev['target_owner'], g['v_player'])} prod={ev['target_prod']} "
            f"守军={ev['target_ships']}，ETA≈{eta_s}，{cap_s}。"
        )
    lines.append("")
    lines.append("**分阶段运营**")
    lines.append("")
    for name, _, _ in PHASES:
        lines.append(phase_line(name, g["phase_stats"][name]))
    lines.append("")
    lines.append("**关键夺点/丢点**")
    lines.append("")
    if g["high_value_gains"]:
        for f in g["high_value_gains"]:
            lines.append(
                f"- 夺点 t={f.step}: P{f.planet} prod={f.production}，"
                f"{fmt_owner(f.prev_owner, g['v_player'])} -> 己方，落点后 {f.ships_after:.0f} ships。"
            )
    else:
        lines.append("- 没有明显的高产或敌方关键夺点；这盘更多靠存量/低产点累积。")
    if g["painful_losses"]:
        for f in g["painful_losses"]:
            lines.append(
                f"- 丢点 t={f.step}: P{f.planet} prod={f.production}，"
                f"己方 -> {fmt_owner(f.new_owner, g['v_player'])}，对方落点后 {f.ships_after:.0f} ships。"
            )
    else:
        lines.append("- 未出现高价值核心星球的严重丢失。")
    repeated = [(pid, c) for pid, c in g["repeated_planets"].most_common(5) if c >= 3]
    if repeated:
        lines.append("- 反复争夺点：" + ", ".join(f"P{pid}({c}次易主)" for pid, c in repeated) + "。")
    lines.append("")
    lines.append("**这盘读法**")
    lines.append("")
    cp80 = next((c for c in g["checkpoints"] if c["step"] >= 80), g["checkpoints"][-1] if g["checkpoints"] else None)
    last = g["checkpoints"][-1] if g["checkpoints"] else None
    opening = g["phase_stats"]["opening_0_50"]
    early = g["phase_stats"]["early_51_120"]
    if g["result"] == "win":
        lines.append(
            f"- 胜利主线：前 50 步已经发起 {opening['launches']} 次动作，"
            f"优先把可消化中立和近身敌点转换成产能；到 t≈80 时产能差为 {cp80['prod_delta'] if cp80 else 'NA'}，"
            f"随后用 {early['launches']} 次早中期动作继续滚雪球。"
        )
        if last:
            lines.append(
                f"- 收官状态：最终船差 {last['score_delta']}，说明优势不是单次大团战，而是持续把星球产能和在途舰队转化为压制。"
            )
    else:
        lines.append(
            f"- 风险暴露：这盘前 50 步动作 {opening['launches']} 次，"
            f"到 t≈80 产能差 {cp80['prod_delta'] if cp80 else 'NA'}、船差 {cp80['score_delta'] if cp80 else 'NA'}；"
            "如果这里落后，Vadasz 也会被更快扩张/更高频压制打穿。"
        )
        if g["painful_losses"]:
            lines.append("- 败因更像是中前期关键星守不住，而不是后期决策突然崩。")
    lines.append("")
    return lines


def write_report(games: list[dict[str, Any]], out: Path) -> None:
    wins = [g for g in games if g["result"] == "win"]
    losses = [g for g in games if g["result"] == "loss"]
    lines: list[str] = []
    lines.append("# Vadasz Replays 逐盘复盘")
    lines.append("")
    lines.append(f"数据源：`data/myreplay/vadasz_replays`，共 {len(games)} 盘；Vadasz 胜 {len(wins)}，负 {len(losses)}。")
    lines.append("")
    lines.append("## 总体结论")
    lines.append("")
    if wins:
        win_open = [g["phase_stats"]["opening_0_50"]["launches"] for g in wins]
        win_early = [g["phase_stats"]["early_51_120"]["launches"] for g in wins]
        win_neutral = [g["phase_stats"]["opening_0_50"]["classes"]["neutral"] for g in wins]
        win_enemy = [g["phase_stats"]["early_51_120"]["classes"]["enemy"] for g in wins]
        lines.append(f"- 胜局开局不是保守攒船：0-50 平均 {mean(win_open):.1f} 次发射，其中打中立平均 {mean(win_neutral):.1f} 次。")
        lines.append(f"- 51-120 步平均 {mean(win_early):.1f} 次发射，早中期开始从纯扩张转为抢敌点/补刀；该阶段打敌方目标平均 {mean(win_enemy):.1f} 次。")
    if losses:
        loss_open = [g["phase_stats"]["opening_0_50"]["launches"] for g in losses]
        loss_early = [g["phase_stats"]["early_51_120"]["launches"] for g in losses]
        lines.append(f"- 败局也会高频行动，0-50 平均 {mean(loss_open):.1f} 次、51-120 平均 {mean(loss_early):.1f} 次；所以差距不只是动作数量，而是目标选择、守点和第三方战局判断。")
    lines.append("- 可迁移的核心打法：小窗口连续出兵、优先低守军高产能点、拿到点后继续从新点外扩，不把舰队长期闲置在后方；领先后压敌高产点，落后时避免在反复易主点继续喂船。")
    lines.append("")
    lines.append("## 下一步工作")
    lines.append("")
    lines.append("1. 把 Vadasz 胜局前 120 步提取成 imitation 数据：源星、目标星、出兵比例、目标类型、ETA、是否成功占领。")
    lines.append("2. 在规则 bot 里加 opening tempo floor：0-50 步若安全目标存在，必须持续从可行动星发射，而不是等到大额守军。")
    lines.append("3. 加目标评分 `production / (守军 + ETA成本)`，并对 prod>=3、守军低、离己方新占点近的目标加权。")
    lines.append("4. 加守点/反打评估：对 2-4 步内将丢的高产点，优先补防；对反复易主点设置止损阈值。")
    lines.append("5. 用本脚本的逐盘指标回归验证：比较我们的 replay 是否在 t=80 产能差、0-50 发射次数、51-120 敌点攻击次数上接近 Vadasz 胜局分布。")
    lines.append("")
    lines.append("## 逐盘复盘")
    lines.append("")
    for g in games:
        lines.extend(describe_game(g))
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default="data/myreplay/vadasz_replays")
    parser.add_argument("--out", default="docs/vadasz_replays_deep_review.md")
    args = parser.parse_args()
    replay_dir = Path(args.replay_dir)
    paths = sorted(replay_dir.glob("*.json"), key=lambda p: p.name)
    games = [summarize_game(p) for p in paths]
    write_report(games, Path(args.out))
    print(f"wrote {args.out} from {len(games)} games")


if __name__ == "__main__":
    main()

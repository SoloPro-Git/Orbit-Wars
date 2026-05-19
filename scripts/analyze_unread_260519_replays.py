from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.expert.action_labeling import infer_target_planet_id


PHASES = [
    ("opening_0_35", 0, 35),
    ("expand_36_80", 36, 80),
    ("fight_81_140", 81, 140),
    ("mid_141_240", 141, 240),
    ("late_241_end", 241, 10_000),
]

CHECKPOINTS = [0, 10, 20, 35, 50, 65, 80, 100, 120, 160, 220, 320, 499]


def fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    ratio = max(0.0, min(1.0, math.log(max(ships, 1.0)) / math.log(1000.0)))
    return 1.0 + 5.0 * (ratio**1.5)


def dist(a: list[Any], b: list[Any]) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def by_id(planets: list[list[Any]]) -> dict[int, list[Any]]:
    return {int(p[0]): p for p in planets}


def planet_count(obs: dict[str, Any], player: int) -> int:
    return sum(1 for p in obs.get("planets", []) if int(p[1]) == player)


def prod(obs: dict[str, Any], player: int) -> int:
    return sum(int(p[6]) for p in obs.get("planets", []) if int(p[1]) == player)


def ships(obs: dict[str, Any], player: int) -> float:
    return float(
        sum(float(p[5]) for p in obs.get("planets", []) if int(p[1]) == player)
        + sum(float(f[6]) for f in obs.get("fleets", []) if int(f[1]) == player)
    )


def phase_name(step: int) -> str:
    for name, lo, hi in PHASES:
        if lo <= step <= hi:
            return name
    return PHASES[-1][0]


def owner_name(owner: int | None, names: list[str], solo: int) -> str:
    if owner is None:
        return "未知"
    if owner == -1:
        return "中立"
    label = "Solo" if owner == solo else names[owner]
    return f"P{owner} {label}"


def target_class(owner: int | None, player: int) -> str:
    if owner is None:
        return "unknown"
    if owner == player:
        return "friendly"
    if owner == -1:
        return "neutral"
    return "enemy"


@dataclass
class ActionEvent:
    step: int
    player: int
    source: int
    target: int | None
    target_owner: int | None
    target_prod: int | None
    target_ships: float | None
    ships: int
    source_prod: int | None
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
    production: int
    ships_after: float


def load_data(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    frames: list[dict[str, Any]] = []
    for step_row in data.get("steps", []):
        if not step_row:
            continue
        obs = step_row[0].get("observation")
        if obs:
            frames.append(obs)
    return data, frames


def nearest_frame(frames: list[dict[str, Any]], wanted: int) -> dict[str, Any]:
    return min(frames, key=lambda o: abs(int(o.get("step", 0)) - wanted))


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


def build_actions(data: dict[str, Any], names: list[str]) -> list[ActionEvent]:
    events: list[ActionEvent] = []
    for step_row in data.get("steps", []):
        for player, agent in enumerate(step_row[: len(names)]):
            obs = agent.get("observation") or {}
            actions = agent.get("action") or []
            if not obs or not actions:
                continue
            step = int(obs.get("step", step_row[0].get("observation", {}).get("step", 0) or 0))
            planets = by_id(obs.get("planets", []))
            outgoing: Counter[int] = Counter()
            for raw in actions:
                if len(raw) >= 3:
                    outgoing[int(raw[0])] += int(raw[2])
            for raw in actions:
                if len(raw) < 3:
                    continue
                source_id = int(raw[0])
                angle = float(raw[1])
                sent = int(raw[2])
                source = planets.get(source_id)
                target_id = infer_target_planet_id(obs, source_id, angle, sent)
                target = planets.get(target_id) if target_id is not None else None
                after = float(source[5]) if source else 0.0
                before = after + float(outgoing[source_id])
                eta = None
                if source is not None and target is not None:
                    eta = dist(source, target) / max(fleet_speed(sent), 1e-6)
                events.append(
                    ActionEvent(
                        step=step,
                        player=player,
                        source=source_id,
                        target=target_id,
                        target_owner=int(target[1]) if target else None,
                        target_prod=int(target[6]) if target else None,
                        target_ships=float(target[5]) if target else None,
                        ships=sent,
                        source_prod=int(source[6]) if source else None,
                        source_ships_before=before,
                        source_ships_after=after,
                        sent_ratio=sent / before if before > 0 else 0.0,
                        eta=eta,
                    )
                )
    return events


def build_flips(frames: list[dict[str, Any]]) -> list[FlipEvent]:
    flips: list[FlipEvent] = []
    if not frames:
        return flips
    prev_planets = by_id(frames[0].get("planets", []))
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
                        production=int(cur[6]),
                        ships_after=float(cur[5]),
                    )
                )
        prev_planets = cur_planets
    return flips


def phase_stats(events: list[ActionEvent], player: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, _, _ in PHASES:
        out[name] = {
            "launches": 0,
            "ships": 0,
            "classes": Counter(),
            "ratios": [],
            "multi_turns": 0,
            "turns": Counter(),
        }
    for ev in events:
        if ev.player != player:
            continue
        st = out[phase_name(ev.step)]
        st["launches"] += 1
        st["ships"] += ev.ships
        st["classes"][target_class(ev.target_owner, player)] += 1
        st["ratios"].append(ev.sent_ratio)
        st["turns"][ev.step] += 1
    for st in out.values():
        st["multi_turns"] = sum(1 for n in st["turns"].values() if n >= 2)
    return out


def summarize_game(path: Path) -> dict[str, Any]:
    data, frames = load_data(path)
    names = [str(n) for n in data.get("info", {}).get("TeamNames", [])]
    solo_candidates = [i for i, n in enumerate(names) if n.lower() == "solo"]
    if not solo_candidates:
        raise ValueError(f"Solo not found in {path}")
    solo = solo_candidates[0]
    rewards = [float(x) for x in data.get("rewards", [])]
    winner = max(range(len(names)), key=lambda i: rewards[i] if i < len(rewards) else -999)
    actions = build_actions(data, names)
    flips = build_flips(frames)

    checkpoints = []
    seen = set()
    for wanted in CHECKPOINTS + ([int(frames[-1].get("step", 0))] if frames else []):
        if not frames:
            break
        obs = nearest_frame(frames, wanted)
        step = int(obs.get("step", 0))
        if step in seen:
            continue
        seen.add(step)
        player_rows = []
        for pid, name in enumerate(names):
            player_rows.append(
                {
                    "player": pid,
                    "name": name,
                    "planets": planet_count(obs, pid),
                    "prod": prod(obs, pid),
                    "ships": round(ships(obs, pid), 1),
                }
            )
        checkpoints.append({"step": step, "players": player_rows})

    first_actions = defaultdict(list)
    for ev in actions:
        if ev.player in {solo, winner} and len(first_actions[ev.player]) < 16:
            cap_step = None
            if ev.target_owner != ev.player:
                cap_step = capture_time(frames, ev.step, ev.target, ev.player)
            first_actions[ev.player].append({**ev.__dict__, "capture_step": cap_step})

    solo_losses = [f for f in flips if f.prev_owner == solo]
    solo_gains = [f for f in flips if f.new_owner == solo]
    winner_gains = [f for f in flips if f.new_owner == winner]
    winner_from_solo = [f for f in winner_gains if f.prev_owner == solo]
    winner_from_others = [f for f in winner_gains if f.prev_owner not in {-1, solo}]
    winner_neutrals = [f for f in winner_gains if f.prev_owner == -1]

    direct_pressure = [
        ev for ev in actions if ev.player == winner and ev.target_owner == solo
    ]
    solo_pressure = [
        ev for ev in actions if ev.player == solo and ev.target_owner == winner
    ]
    repeated = Counter(f.planet for f in flips)

    return {
        "file": path.name,
        "episode": data.get("info", {}).get("EpisodeId", path.stem),
        "seed": data.get("info", {}).get("seed"),
        "names": names,
        "players": len(names),
        "rewards": rewards,
        "solo": solo,
        "winner": winner,
        "length": int(frames[-1].get("step", 0)) if frames else len(data.get("steps", [])) - 1,
        "checkpoints": checkpoints,
        "phase_solo": phase_stats(actions, solo),
        "phase_winner": phase_stats(actions, winner),
        "first_actions": dict(first_actions),
        "solo_losses": solo_losses,
        "solo_gains": solo_gains,
        "winner_gains": winner_gains,
        "winner_from_solo": winner_from_solo,
        "winner_from_others": winner_from_others,
        "winner_neutrals": winner_neutrals,
        "direct_pressure": direct_pressure,
        "solo_pressure": solo_pressure,
        "repeated": repeated,
    }


def stats_line(label: str, stats: dict[str, Any]) -> str:
    cls = stats["classes"]
    ratio = mean(stats["ratios"]) if stats["ratios"] else 0.0
    return (
        f"- `{label}`: {stats['launches']}次 / {stats['ships']}船；"
        f"打中立{cls['neutral']}、打敌{cls['enemy']}、支援{cls['friendly']}、未知{cls['unknown']}；"
        f"平均出兵比例 {ratio:.0%}；多动作回合 {stats['multi_turns']}。"
    )


def checkpoint_table(g: dict[str, Any]) -> list[str]:
    lines = ["| t | Solo 星/产/船 | 胜者 星/产/船 | 差值(Solo-胜者) |", "|---:|---:|---:|---:|"]
    solo = g["solo"]
    winner = g["winner"]
    for cp in g["checkpoints"]:
        rows = {r["player"]: r for r in cp["players"]}
        s = rows[solo]
        w = rows[winner]
        lines.append(
            f"| {cp['step']} | {s['planets']}/{s['prod']}/{s['ships']:.0f} | "
            f"{w['planets']}/{w['prod']}/{w['ships']:.0f} | "
            f"{s['planets']-w['planets']}/{s['prod']-w['prod']}/{s['ships']-w['ships']:.0f} |"
        )
    return lines


def format_action(ev: dict[str, Any], g: dict[str, Any]) -> str:
    eta = "?" if ev["eta"] is None else f"{ev['eta']:.1f}"
    target = "?" if ev["target"] is None else f"P{ev['target']}"
    cap = ev.get("capture_step")
    if ev["target_owner"] == ev["player"]:
        cap_text = "，支援/续压己方点"
    else:
        cap_text = f"，t={cap} 到手" if cap is not None else "，未稳定到手"
    return (
        f"- t={ev['step']}: P{ev['source']}(prod={ev['source_prod']}) -> {target}，"
        f"发 {ev['ships']}/{ev['source_ships_before']:.0f}({ev['sent_ratio']:.0%})，"
        f"发后 {ev['source_ships_after']:.0f}；目标 {owner_name(ev['target_owner'], g['names'], g['solo'])} "
        f"prod={ev['target_prod']} 守={ev['target_ships']}，ETA≈{eta}{cap_text}。"
    )


def flip_line(f: FlipEvent, g: dict[str, Any]) -> str:
    return (
        f"- t={f.step}: P{f.planet} prod={f.production}，"
        f"{owner_name(f.prev_owner, g['names'], g['solo'])} -> {owner_name(f.new_owner, g['names'], g['solo'])}，"
        f"落点后 {f.ships_after:.0f} 船。"
    )


def first_cp_at_or_after(g: dict[str, Any], step: int) -> dict[str, Any] | None:
    return next((cp for cp in g["checkpoints"] if cp["step"] >= step), g["checkpoints"][-1] if g["checkpoints"] else None)


def diagnose(g: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    solo = g["solo"]
    winner = g["winner"]
    cp50 = first_cp_at_or_after(g, 50)
    cp80 = first_cp_at_or_after(g, 80)
    cp120 = first_cp_at_or_after(g, 120)
    for label, cp in [("t≈50", cp50), ("t≈80", cp80), ("t≈120", cp120)]:
        if not cp:
            continue
        rows = {r["player"]: r for r in cp["players"]}
        s = rows[solo]
        w = rows[winner]
        if s["prod"] + 8 <= w["prod"]:
            lines.append(f"- {label} 已经出现产能硬差：Solo {s['prod']} vs 胜者 {w['prod']}，后面基本是在还这笔利息。")
            break
    first_loss = g["solo_losses"][0] if g["solo_losses"] else None
    if first_loss:
        lines.append(
            f"- 第一颗丢星在 t={first_loss.step}：P{first_loss.planet} prod={first_loss.production}，"
            f"对方落点后 {first_loss.ships_after:.0f} 船；这通常是本局从扩张转为失血的转折。"
        )
    solo_open = sum(g["phase_solo"][p]["launches"] for p in ["opening_0_35", "expand_36_80"])
    win_open = sum(g["phase_winner"][p]["launches"] for p in ["opening_0_35", "expand_36_80"])
    if win_open >= solo_open * 1.5 + 8:
        lines.append(f"- 前 80 步发射节奏被拉开：Solo {solo_open} 次，胜者 {win_open} 次。对手赢在持续把新增产能立刻再投资。")
    direct = len([ev for ev in g["direct_pressure"] if ev.step <= 120])
    if direct:
        lines.append(f"- 胜者前 120 步直接瞄 Solo 星球 {direct} 次；不是纯扩张胜，而是扩张后很快开始割 Solo 边境。")
    third_party = len([f for f in g["winner_from_others"] if f.step <= 160])
    if g["players"] == 4 and third_party:
        lines.append(f"- 4p 里胜者还吃了第三方战果：t<=160 从其他玩家手里拿 {third_party} 颗，这解释了为什么 Solo 有时不是被胜者第一时间打死，却仍输给胜者滚雪球。")
    if not lines:
        lines.append("- 这局不是单一灾难点，而是多个小差距叠加：目标质量、守点、行动频率都略低，最后在中盘被放大。")
    return lines


def describe_game(g: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    names = g["names"]
    solo = g["solo"]
    winner = g["winner"]
    lines.append(
        f"## {g['file']}：Solo P{solo} 负于 P{winner} {names[winner]}，"
        f"{g['players']}p，长度 t={g['length']}"
    )
    lines.append("")
    lines.append(f"- 奖励：`{g['rewards']}`；seed：`{g['seed']}`。")
    lines.append(f"- 玩家：{', '.join(f'P{i} {n}' for i, n in enumerate(names))}。")
    lines.append("")
    lines.append("### 时间线")
    lines.extend(checkpoint_table(g))
    lines.append("")
    lines.append("### 发射节奏对比")
    lines.append("")
    lines.append("Solo：")
    for name, _, _ in PHASES:
        lines.append(stats_line(name, g["phase_solo"][name]))
    lines.append("")
    lines.append(f"胜者 {names[winner]}：")
    for name, _, _ in PHASES:
        lines.append(stats_line(name, g["phase_winner"][name]))
    lines.append("")
    lines.append("### 开局逐手")
    lines.append("")
    lines.append("Solo 前 16 手：")
    for ev in g["first_actions"].get(solo, []):
        lines.append(format_action(ev, g))
    lines.append("")
    lines.append(f"胜者 {names[winner]} 前 16 手：")
    for ev in g["first_actions"].get(winner, []):
        lines.append(format_action(ev, g))
    lines.append("")
    lines.append("### 星球易主细节")
    lines.append("")
    lines.append("Solo 关键丢点：")
    for f in g["solo_losses"][:18]:
        lines.append(flip_line(f, g))
    if not g["solo_losses"]:
        lines.append("- 无。")
    lines.append("")
    lines.append("Solo 关键拿点/反打：")
    for f in g["solo_gains"][:14]:
        lines.append(flip_line(f, g))
    if not g["solo_gains"]:
        lines.append("- 无。")
    lines.append("")
    lines.append(f"胜者 {names[winner]} 的赢法拆解：")
    lines.append(
        f"- 总拿点 {len(g['winner_gains'])}：中立 {len(g['winner_neutrals'])}，"
        f"从 Solo 手里 {len(g['winner_from_solo'])}，从其他玩家手里 {len(g['winner_from_others'])}。"
    )
    if g["winner_from_solo"]:
        lines.append("- 直接吃 Solo：")
        for f in g["winner_from_solo"][:12]:
            lines.append(flip_line(f, g))
    if g["winner_from_others"]:
        lines.append("- 吃第三方/乱战收益：")
        for f in g["winner_from_others"][:10]:
            lines.append(flip_line(f, g))
    if g["winner_neutrals"]:
        high_neutrals = [f for f in g["winner_neutrals"] if f.production >= 3][:12]
        if high_neutrals:
            lines.append("- 关键高产中立：")
            for f in high_neutrals:
                lines.append(flip_line(f, g))
    repeated = [(pid, n) for pid, n in g["repeated"].most_common(8) if n >= 3]
    if repeated:
        lines.append("- 反复争夺点：" + ", ".join(f"P{pid}({n}次)" for pid, n in repeated) + "。")
    lines.append("")
    lines.append("### 细粒度诊断")
    lines.extend(diagnose(g))
    lines.append("")
    return lines


def write_report(games: list[dict[str, Any]], out: Path, replay_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# unread_260519 失败录像逐局细复盘")
    lines.append("")
    lines.append(f"数据源：`{replay_dir}`，共 {len(games)} 盘，Solo 全部失败。")
    lines.append("")
    two_p = [g for g in games if g["players"] == 2]
    four_p = [g for g in games if g["players"] == 4]
    lines.append("## 总体规律")
    lines.append("")
    for label, subset in [("2p", two_p), ("4p", four_p), ("全部", games)]:
        if not subset:
            continue
        solo_80 = []
        win_80 = []
        solo_launch_80 = []
        win_launch_80 = []
        first_losses = []
        for g in subset:
            cp = first_cp_at_or_after(g, 80)
            if cp:
                rows = {r["player"]: r for r in cp["players"]}
                solo_80.append(rows[g["solo"]]["prod"])
                win_80.append(rows[g["winner"]]["prod"])
            solo_launch_80.append(sum(g["phase_solo"][p]["launches"] for p in ["opening_0_35", "expand_36_80"]))
            win_launch_80.append(sum(g["phase_winner"][p]["launches"] for p in ["opening_0_35", "expand_36_80"]))
            if g["solo_losses"]:
                first_losses.append(g["solo_losses"][0].step)
        lines.append(
            f"- {label}: t≈80 Solo 平均产能 {mean(solo_80):.1f}，胜者 {mean(win_80):.1f}；"
            f"前80步发射 Solo {mean(solo_launch_80):.1f} 次，胜者 {mean(win_launch_80):.1f} 次；"
            f"首个丢点平均 t={mean(first_losses):.1f}。"
        )
    lines.append("")
    lines.append("核心结论：这批不是“最后一波打输了”，而是前 80-120 步被对手用更高行动频率、更快高产中立转换、以及更果断的边境收割滚开。4p 里还经常出现胜者不一定最早打 Solo，但它会吃第三方和 Solo 崩盘后的真空区。")
    lines.append("")
    lines.append("## 下一轮改法")
    lines.append("")
    lines.append("- `4p` 优先修 launch throughput：前 80 步 Solo 平均只有 14.8 次发射，胜者 82.6 次。需要 no-action fallback / reserve 放松 / 候选拓宽，尤其在 t=35..120。")
    lines.append("- 加早期失血报警：首个丢点平均 t=54.2，很多局第一颗丢星后 30 回合内连锁崩。触发条件可以是 `t<90` 丢星、敌方连续瞄准己方、或 15 回合内净丢 2 星。")
    lines.append("- 高产点/前线点不要只靠事后 recapture：2p 里经常 t=50..120 产能还接近，但对手直接瞄 Solo 20-80 次，把边境打穿。需要 arrival-based reinforcement 和 multi-source hold。")
    lines.append("- 4p 要识别“胜者吃第三方”而不是只看谁正在打我：多盘胜者大量吃其他玩家或乱战尾刀，Solo 如果只做局部防守，会被地图控制者滚死。")
    lines.append("- 把本报告里的 t≈50/80/120 产能差、前80发射数、首个丢点时间做成回归指标；后续规则改动先看这些硬指标有没有往胜者分布靠近。")
    lines.append("")
    lines.append("## 逐局复盘")
    lines.append("")
    for g in games:
        lines.extend(describe_game(g))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default="data/myreplay/unread_260519")
    parser.add_argument("--out", default="docs/unread_260519_deep_failure_review.md")
    args = parser.parse_args()
    replay_dir = Path(args.replay_dir)
    games = [summarize_game(p) for p in sorted(replay_dir.glob("*.json"), key=lambda p: p.name)]
    write_report(games, Path(args.out), replay_dir)
    print(f"wrote {args.out} from {len(games)} games")


if __name__ == "__main__":
    main()

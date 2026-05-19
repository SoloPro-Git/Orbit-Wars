# unread replay strategy analysis and plan/spec

Source: `data/myreplay/unread/*.json`

Scope: 20 losses containing `Solo`: 8 two-player games and 12 four-player games.

## Executive summary

Solo is usually losing before the final attack phase. The common failure is not one
bad battle; it is a tempo gap that opens in the first 80 turns.

Across the 20 games:

- Solo early launches, `t <= 80`: mean 32.4 launches / 760.9 ships.
- Winner early launches, `t <= 80`: mean 75.5 launches / 1990.7 ships.
- Solo total launches: mean 93.5 launches / 2342.6 ships.
- Winner total launches: mean 335.7 launches / 11814.0 ships.
- First Solo planet loss: mean turn 48.9, median turn 48.
- At `t=50`, Solo averages 16.1 production and 5.8 planets; winners average 22.1
  production and 8.1 planets.
- At `t=80`, Solo averages 17.9 production and 7.1 planets; winners average 30.5
  production and 12.3 planets.
- At `t=120`, Solo averages only 15.0 production and 6.1 planets; winners average
  39.2 production and 15.8 planets.

The winning bots repeatedly use three patterns:

1. Fast neutral conversion, including low-garrison `production >= 3` planets.
2. Continuous small/medium pressure against border planets starting around turns
   45-70.
3. High launch cadence after the first expansion, recycling production immediately
   instead of banking ships on rear planets.

Solo repeatedly shows four weaknesses:

1. Under-launching compared with top opponents, especially after the first few
   neutral captures.
2. Weak border defense: newly captured planets are often taken back within 10-30
   turns.
3. Poor response to opponent snowball: Solo does not switch into emergency
   consolidation or counter-punch when enemy production passes Solo by 10+.
4. In four-player games, Solo gets pinched by adjacent players and then the final
   winner harvests the collapsed quadrant.

## Strategy spec

### S1: Early expansion target scoring

Goal: by turn 50, match or exceed the future winner's planet count and production.

Implement an early expansion scorer for `t < 70`:

- Score neutral targets by `production / required_ships / travel_time`.
- Add a strong bonus for `production >= 3` and neutral ships below 15.
- Add a bonus for symmetric mirror targets that the opponent is likely to take.
- Penalize routes crossing near the sun and targets inside obvious enemy arrival
  windows.
- Allow multi-source capture when one source cannot take a high-value planet fast.

Acceptance checks:

- On the 20 unread replays, an offline policy audit should recommend at least as
  many neutral captures as Solo made before turn 50.
- In fast simulator self-play, average owned production at `t=50` should improve
  against the current policy without increasing eliminations before `t=80`.

### S2: Launch cadence and ship budget

Goal: stop banking idle ships while enemies are converting the map.

Rules:

- During `t < 100`, keep only a local reserve on non-front planets:
  `reserve = max(6, incoming_enemy_power + 2)`.
- Any ships above reserve should be assigned to expansion, reinforcement, or a
  planned attack within 3 turns.
- Front planets may keep higher reserves only if enemy fleets or enemy high-prod
  planets are nearby.
- If total launch count in the last 20 turns is below a tempo floor, force a
  productive action from the safest rich planet.

Initial tempo floors:

- Two-player: at least 15 launches by turn 50, 45 launches by turn 80.
- Four-player: at least 12 launches by turn 50, 35 launches by turn 80, unless
  boxed in by hostile fleets.

### S3: Border defense and recapture

Goal: newly won planets should not be cheap gifts.

For every owned planet:

- Compute enemy arrival threat within the next 20 turns.
- If enemy can flip it with a small surplus, either reinforce from nearest safe
  owned planet or evacuate only if holding is negative value.
- When a Solo planet is lost with low remaining enemy garrison, schedule an
  immediate recapture if travel time and enemy reinforcements allow it.
- Detect repeated flips on the same planet and stop trickling ships into losing
  trades.

Acceptance checks:

- In replay audit mode, flag every Solo planet loss before turn 80 and produce a
  reinforcement/recapture action candidate.
- Reduce median first planet loss from turn 48 to later than turn 65 in local
  two-player regression matches.

### S4: Opponent snowball response

Goal: when one player becomes leader, stop letting them harvest isolated planets.

Trigger:

- Enemy production exceeds Solo by 8 before turn 80, or by 12 before turn 140.
- Enemy planet count exceeds Solo by 5.
- Solo loses two planets within 15 turns.

Response:

- Switch from greedy neutral expansion to consolidation.
- Prefer attacks on the leader's low-garrison high-production planets.
- Reinforce a compact cluster instead of holding far isolated planets.
- In four-player games, avoid feeding ships into wars with non-leaders unless they
  are about to eliminate Solo.

### S5: Training/data plan

Use these unread losses as hard-negative evaluation cases, not as imitation data.

Build a replay audit script that emits:

- `first_solo_loss_turn`
- production and planet count at `t=20,50,80,120`
- launch count and launched ships by phase
- first 10 neutral captures by Solo and winner
- Solo planets lost before turn 100
- whether the policy would choose expansion, defense, recapture, or leader attack

Then add regression scenarios:

- Two-player high-tempo mirrors from episodes `76934632`, `76937889`,
  `76938253`, `76939868`.
- Four-player pinch scenarios from episodes `76934377`, `76937624`, `76940742`.
- Slow-map long games from episodes `76938983`, `76939353`, `76940240`,
  `76940470`.

## Per-episode analysis

### episode-76932660: restricteur beat Solo, 2p, 156 turns

Winner strategy:

- restricteur started later than Solo in first action timing, but converted much
  more production by midgame.
- By `t=80`, restricteur had 9 planets / 14 production; Solo had 3 planets / 6
  production.
- restricteur launched 71 times for 2391 ships; Solo launched 25 times for 442.
- The winner began stripping Solo planets from `t=65`, then chained captures every
  10-15 turns.

Solo failure:

- Solo captured early neutrals but did not defend them. Planet 27 was captured at
  `t=18` and lost at `t=82`; planet 6 was captured at `t=48` and lost at `t=103`.
- Production collapsed from 6 at `t=80` to 3 at `t=120`, while restricteur reached
  33.
- This is a pure launch-cadence and border-defense loss.

### episode-76933032: Felipe Ferreira beat Solo, 4p, 252 turns

Winner strategy:

- Felipe kept stable production growth while other players fought.
- By `t=120`, Felipe had 7 planets / 15 production; Solo had only the home planet
  / 1 production.
- Felipe's total pressure was huge: 127 launches / 5905 ships.

Solo failure:

- Solo opened decently, but lost planets 8, 4, 1, and 12 between `t=48` and
  `t=74`.
- Solo's production fell from 9 at `t=50` to 2 at `t=80`.
- Schanitater did the early damage to Solo; Felipe then snowballed from the
  cleaner position. Solo needs pinch detection and emergency consolidation.

### episode-76934377: Alvaro beat Solo, 4p, 210 turns

Winner strategy:

- Alvaro expanded fastest: 9 planets / 26 production at `t=50`, already ahead of
  Solo's 6 planets / 15 production.
- Alvaro kept attacking owned planets, not just neutrals, and reached 44
  production by `t=120`.
- Alvaro launched 202 times, nearly double Solo's 106, with strong midgame
  follow-through.

Solo failure:

- Solo's early captures were respectable, but several were not held: planets 14,
  18, and 26 were lost by `t=51`.
- Solo reached 31 production at `t=120`, but was already behind Alvaro's 44 and
  exposed to repeated captures.
- The main issue is not expansion start; it is post-capture defense and attacking
  the leader before the leader reaches critical mass.

### episode-76934450: Oleh Patsan beat Solo, 4p, 118 turns

Winner strategy:

- Oleh snowballed extremely fast: 25 production at `t=50`, 44 at `t=80`.
- The winner launched 137 times / 6900 ships in a short game.
- Oleh directly took Solo planets from turn 27 onward.

Solo failure:

- Solo had only 2 planets / 7 production at `t=50` and was eliminated by `t=80`.
- Multiple Solo planets flipped repeatedly between `t=27` and `t=54`.
- Solo under-launched badly: 11 launches total vs winner's 137.

### episode-76934632: Caiden Matthews beat Solo, 2p, 109 turns

Winner strategy:

- Caiden captured high-production neutrals early and converted the map by brute
  tempo.
- By `t=50`, Caiden had 16 planets / 47 production; Solo had 12 planets / 37.
- By `t=80`, Caiden had 26 planets / 68 production; Solo had fallen to 9 planets
  / 27.
- Caiden launched 131 times / 5818 ships; Solo launched 61 / 1501.

Solo failure:

- Solo's opening was not terrible, but Caiden's follow-up was much stronger.
- Solo began losing production planets at `t=54`, then lost planet 12 at `t=58`,
  planet 1 at `t=59`, and multiple others through `t=73`.
- Need stronger defense trigger when opponent's production lead exceeds 10.

### episode-76934678: Anirudh K beat Solo, 2p, 346 turns

Winner strategy:

- Anirudh out-expanded in the first 80 turns: 20 planets / 42 production vs Solo's
  15 planets / 37.
- The winner launched 584 times / 24977 ships, using constant pressure for a long
  squeeze.
- After `t=250`, Anirudh widened production from 44 vs 32 to 76 vs 0 by the end.

Solo failure:

- Solo stayed competitive until about `t=160`, but could not stop the gradual
  loss of border planets.
- First important losses started at `t=76`, then `t=93`, `t=95`, `t=101`,
  `t=108`, `t=109`.
- This calls for better recapture logic and sustained launch cadence, not just a
  stronger opening.

### episode-76935872: currypurin beat Solo, 4p, 500 turns

Winner strategy:

- currypurin won the map early and then banked an enormous score.
- At `t=80`, currypurin had 15 planets / 37 production; Solo had 3 planets / 7.
- By `t=120`, currypurin had 27 planets / 69 production and Solo had one planet.

Solo failure:

- Solo under-expanded hard: only 3 planets at `t=50` and `t=80`.
- Solo's planets were taken from `t=20` onward, including planet 30 immediately
  after capture.
- The policy needs to avoid isolated low-value expansion and prioritize holding a
  compact production cluster.

### episode-76936394: Thomas beat Solo, 4p, 163 turns

Winner strategy:

- Thomas stayed alive through early fights and then harvested the map after other
  players weakened each other.
- Thomas reached 36 production at `t=80`, 59 at `t=120`, and 90 at the end.
- Thomas used owned-planet attacks heavily from `t=30`.

Solo failure:

- Solo's early expansion was okay: 7 planets / 22 production at `t=50`.
- But Solo lost high-value contested planets from `t=32` and had zero production
  by `t=120`.
- Solo was overexposed in a four-player fight and did not recognize when to
  consolidate or abandon losing border trades.

### episode-76936647: automatylicza beat Solo, 4p, 214 turns

Winner strategy:

- automatylicza and schanitater had very high launch cadence; automatylicza later
  converted the advantage.
- Winner launched 486 times / 18836 ships; Solo launched only 29 / 617.
- By `t=120`, winner had 10 planets / 28 production; Solo had one planet / 4.

Solo failure:

- Solo mirrored the first few neutral captures but stopped converting after
  turn 30.
- Solo's planets were taken starting `t=48`, and production hit zero by `t=160`.
- This is one of the clearest under-launch failures.

### episode-76937624: c-number beat Solo, 4p, 149 turns

Winner strategy:

- c-number built a compact lead, reaching 10 planets / 26 production at `t=80`.
- The winner attacked Solo and other players' weak border planets starting around
  `t=42`.
- c-number reached 64 production by elimination.

Solo failure:

- Solo had 6 planets / 17 production at `t=50`, but only 4 planets / 12
  production at `t=80`.
- Solo lost planets at `t=42`, `t=53`, `t=55`, `t=56`, `t=60`, and `t=71`.
- Need stronger defense of fresh captures and better evaluation of contested
  border planets.

### episode-76937889: SonOfNike beat Solo, 2p, 178 turns

Winner strategy:

- SonOfNike captured many low-garrison production-5 neutrals early.
- By `t=120`, SonOfNike had 18 planets / 46 production; Solo had 10 planets / 22.
- The winner repeatedly took Solo planets with meaningful garrisons from `t=58`.

Solo failure:

- Solo was competitive at `t=50`, but enemy production was already higher.
- Solo lost planet 15 with 46 ships at `t=71`, planet 14 with 71 ships at `t=89`,
  and several more high-garrison planets after.
- This suggests poor incoming-threat prediction: Solo is holding ships on planets
  but not reinforcing or countering correctly.

### episode-76938253: HassenHamdi Winning InchaAllah beat Solo, 2p, 139 turns

Winner strategy:

- HassenHamdi had a much higher action rate: 458 launches / 10076 ships.
- By `t=80`, winner had 18 planets / 43 production; Solo had 13 planets / 32.
- Winner then took six Solo planets between `t=47` and `t=80`.

Solo failure:

- Solo expanded well enough early, but could not hold the border.
- Planet 20 was lost at `t=62` with 64 remaining enemy ships; planet 12 was lost
  again at `t=75` with 66.
- Need arrival-window defense and immediate recapture planning.

### episode-76938619: HilalElusive beat Solo, 2p, 252 turns

Winner strategy:

- HilalElusive used extreme launch cadence: 1072 launches / 38653 ships.
- The game was equal in production at `t=80` and `t=120`, but Hilal kept
  initiative and slowly removed Solo planets.
- By `t=200`, Hilal had 49 production vs Solo's 23.

Solo failure:

- Solo matched production until the midgame but lost the repeated border fights.
- First losses: `t=66`, `t=69`, `t=85`, `t=86`, `t=93`, `t=106`.
- The issue is long-horizon pressure handling: Solo needs to keep launching and
  recapturing, not settle into passive defense.

### episode-76938983: Alexey Yurasov beat Solo, 2p, 309 turns

Winner strategy:

- Slow opening, then Alexey pulled ahead by taking many small planets.
- By `t=80`, Alexey had 11 planets / 13 production; Solo had 4 planets / 6.
- By `t=250`, Alexey had 24 planets / 42 production; Solo had 4 planets / 10.

Solo failure:

- Solo's expansion rate was too slow on a low-production map.
- Early launch counts were similar, but Solo's target selection produced fewer
  planets and lower production.
- Need low-prod-map logic: cheap neutrals matter because planet count and launch
  bases become the advantage.

### episode-76939353: Fishman97 beat Solo, 4p, 500 turns

Winner strategy:

- Fishman97 survived with few planets early but chose effective attacks on owned
  planets and kept a durable score lead.
- Fishman97 had 18 production by `t=80`, 37 by `t=160`, and banked to 14419 ships
  by the end.
- The winner did not need maximum early neutral count; they needed profitable
  fights and survival.

Solo failure:

- Solo had only 2 planets / 4 production at `t=80`, then never recovered.
- Solo lost planets at `t=58`, `t=59`, `t=60`, `t=80`, and repeated losses after.
- Solo needs a four-player survival mode: when boxed in, stop feeding isolated
  captures and protect one expandable cluster.

### episode-76939621: Sitoa beat Solo, 4p, 164 turns

Winner strategy:

- Sitoa took large, high-value early targets, including a production-5 planet at
  `t=14`.
- By `t=120`, Sitoa had 16 planets / 47 production; Solo had 4 planets / 13.
- Sitoa launched 338 times / 6847 ships, far above Solo's 61 / 1948.

Solo failure:

- Solo opened with reasonable neutral captures, but lost planet 23 at `t=49`,
  planet 15 at `t=55`, and several more through `t=92`.
- At `t=80`, Solo still had 18 production, but Sitoa's pressure and captures
  converted that into collapse.
- Need high-value target contesting and defense of production-4/5 planets.

### episode-76939868: Blu3s beat Solo, 2p, 144 turns

Winner strategy:

- Blu3s expanded aggressively and kept 126 launches by `t=80`.
- By `t=80`, Blu3s had 17 planets / 53 production; Solo had 10 planets / 34.
- Blu3s then took many Solo planets in a tight sequence from `t=56` to `t=71`.

Solo failure:

- Solo was not far behind at `t=50`, but did not respond to the turn-56 attack
  wave.
- Lost planet 15 at `t=62` with 73 remaining enemy ships, indicating a major
  defensive miss.
- Need attack-wave detection and reinforcement from nearby reserves.

### episode-76940240: galaxy2025 beat Solo, 4p, 334 turns

Winner strategy:

- galaxy2025 used enormous launch cadence: 1509 launches / 59843 ships.
- By `t=50`, galaxy2025 had 7 planets / 22 production; Solo had 4 planets / 12.
- The winner steadily absorbed collapsed players and beat Solo in the final
  two-player phase.

Solo failure:

- Solo was still alive and competitive through `t=200`, but fell behind in
  production and action volume.
- Solo lost planets early at `t=37`, `t=63`, `t=87`, `t=88`, `t=91`, `t=93`.
- In long games, Solo needs continuous pressure and leader-targeting; otherwise a
  high-cadence bot will eventually own the map.

### episode-76940470: Felipe Ferreira beat Solo, 4p, 335 turns

Winner strategy:

- Felipe and Solo were close for a long time, but Felipe had better conversion of
  attacks into production.
- At `t=300`, Felipe had 13 planets / 19 production; Solo had 7 planets / 9.
- Felipe launched 211 times / 5879 ships vs Solo's 153 / 3399.

Solo failure:

- Solo lost several planets between `t=69` and `t=100`, which erased the opening
  advantage.
- The map was low-production; each base mattered, and Solo allowed too many cheap
  flips.
- Need low-prod-map recapture and base-count preservation.

### episode-76940742: higaki beat Solo, 4p, 189 turns

Winner strategy:

- higaki did not have the highest early launch count, but made profitable attacks
  and reached 21 production by `t=80`.
- By `t=160`, higaki had 20 planets / 48 production; Solo had 7 planets / 12.
- higaki harvested both Solo and other players after they weakened each other.

Solo failure:

- Solo had 9 planets / 21 production at `t=80`, equal to higaki's production, but
  lost key planets at `t=36`, `t=45`, `t=68`, `t=75`, `t=79`, `t=82`, `t=84`.
- The collapse came from losing contested planets faster than replacing them.
- Need a four-player danger model that identifies when multiple neighbors can hit
  the same quadrant and switches from expansion to fortification.

## Implementation priority

1. Add replay audit tooling and make the above metrics automatic.
2. Improve early neutral target scoring.
3. Add incoming-threat defense and immediate recapture candidates.
4. Add launch tempo floors and idle-ship budget rules.
5. Add leader/pinch detection for four-player games.
6. Validate with fast simulator first, then run the official compatibility check
   only if simulator rules or collision behavior are changed.

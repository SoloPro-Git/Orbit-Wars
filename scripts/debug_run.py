"""本地调试脚本 - 运行游戏并生成 HTML 回放"""
from kaggle_environments import make

env = make("orbit_wars", configuration={"seed": 42}, debug=True)

# 替换第二个 agent 为你想要对战的策略
# "random" / "greedy" / 另一个 main.py
# env.run(["main.py", "random"])
env.run(["random", "random"])

# 查看结果
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# 生成 HTML 回放文件
html = env.render(mode="html")
with open("replay.html", "w") as f:
    f.write(html)

print("Replay saved to replay.html")

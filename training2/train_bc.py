"""Behavior cloning pretrain on rulebase-generated candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training2.batching import collate_rows
from training2.model import CandidatePolicyValueNet


class JsonlDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "planets": torch.tensor(row["planets"], dtype=torch.float32),
            "global": torch.tensor(row["global"], dtype=torch.float32),
            "candidates": torch.tensor(row["candidates"], dtype=torch.float32),
            "mask": torch.tensor(row["candidate_mask"], dtype=torch.float32),
            "target": torch.tensor(row["target"], dtype=torch.long),
            "value": torch.tensor(row["value"], dtype=torch.float32),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="training2/checkpoints/bc.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = JsonlDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_rows,
    )
    model = CandidatePolicyValueNet().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0
        for batch in loader:
            planets = batch["planets"].to(args.device)
            glob = batch["global"].to(args.device)
            candidates = batch["candidates"].to(args.device)
            mask = batch["mask"].to(args.device)
            target = batch["target"].to(args.device)
            value_target = batch["value"].to(args.device)

            logits, value = model(planets, glob, candidates, mask)
            policy_loss = F.cross_entropy(logits, target)
            value_loss = F.mse_loss(value, value_target)
            entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = target.numel()
            total_loss += float(loss.item()) * bs
            total_acc += float((logits.argmax(-1) == target).float().sum().item())
            total_n += bs
        print(f"epoch={epoch} loss={total_loss/total_n:.4f} acc={total_acc/total_n:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

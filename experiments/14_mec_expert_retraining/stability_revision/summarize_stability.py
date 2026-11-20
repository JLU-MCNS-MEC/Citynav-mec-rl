#!/usr/bin/env python3
"""Stage 14-R2: aggregate stability results for TS / TM / Large and draw
comparison figures. Reads only stability_revision/runs/."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ["ts", "tm", "large"]
NAMES = {"ts": "Transformer-S", "tm": "Transformer-M", "large": "Large"}
COLORS = {"ts": "#1f77b4", "tm": "#d62728", "large": "#2ca02c"}


def run_dir(key: str) -> Path:
    return ROOT / "runs" / key / "seed42"


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def load_trace(key: str) -> list[dict]:
    p = run_dir(key) / "grad_trace.jsonl"
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summarize(key: str) -> dict:
    d = run_dir(key)
    hist = load_json(d / "training_history.json", [])
    trace = load_trace(key)
    gns = np.asarray([t["grad_norm_preclip"] for t in trace
                      if isinstance(t.get("grad_norm_preclip"), (int, float))
                      and math.isfinite(t["grad_norm_preclip"])])
    skips = sum(1 for t in trace if t.get("skipped"))
    per_epoch = []
    for h in hist:
        per_epoch.append({
            "epoch": h["epoch"], "train_loss": h["train_loss"],
            "train_acc": h["train_acc"], "dev_macro_f1": h["dev"]["macro_f1"],
            "dev_acc": h["dev"]["accuracy"],
            "pred": [h["dev"]["per_class"][c]["predicted"]
                     for c in ["Stop", "Forward", "TurnLeft", "TurnRight"]],
            "seconds": h["seconds"], "lr": h["lr"],
            "skipped_updates": h.get("skipped_updates", 0),
        })
    first = gns[: min(500, len(gns))] if len(gns) else np.array([np.nan])
    last = gns[-min(500, len(gns)):] if len(gns) else np.array([np.nan])
    return {
        "model": NAMES[key], "key": key,
        "epochs_run": len(hist),
        "params": load_json(d / "DONE.json", {}).get("n_params"),
        "grad_norm": {
            "n_updates": int(len(gns)),
            "first500_median": float(np.nanmedian(first)),
            "last500_median": float(np.nanmedian(last)),
            "max": float(np.nanmax(gns)) if len(gns) else None,
            "skipped_updates": skips,
        },
        "epochs": per_epoch,
        "done": load_json(d / "DONE.json", None),
        "aborted": load_json(d / "ABORT.json", None),
    }


def plot_all(summaries: dict[str, dict]) -> list[str]:
    made = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("WARN matplotlib unavailable:", e)
        return made
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    def save(fig, name):
        p = fig_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(str(p))

    # 1) grad norm per update (log scale)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in MODELS:
        tr = load_trace(k)
        xs = [t["step"] for t in tr if isinstance(t.get("grad_norm_preclip"), (int, float))]
        ys = [max(float(t["grad_norm_preclip"]), 1e-8) for t in tr
              if isinstance(t.get("grad_norm_preclip"), (int, float))]
        if xs:
            ax.plot(xs, ys, lw=0.8, color=COLORS[k], label=NAMES[k])
    ax.set_yscale("log"); ax.set_xlabel("optimizer step")
    ax.set_ylabel("grad norm (pre-clip)"); ax.grid(alpha=.3); ax.legend()
    save(fig, "grad_norm_preclip.png")

    # 2) train loss / dev macro-F1
    for field, ylabel, fname in [("train_loss", "train loss", "train_loss.png"),
                                 ("dev_macro_f1", "Dev Macro-F1", "dev_macro_f1.png")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in MODELS:
            eps = summaries[k]["epochs"]
            if eps:
                ax.plot([e["epoch"] for e in eps], [e[field] for e in eps],
                        marker="o", color=COLORS[k], label=NAMES[k])
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=.3); ax.legend()
        save(fig, fname)

    # 3) token L2 norm after normalization (epoch 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = ["cls", "current", "target", "history", "state", "landmark", "relation"]
    width = 0.8 / max(len(MODELS), 1)
    for i, k in enumerate(MODELS):
        tok = load_json(run_dir(k) / "token_scale_diagnostics_epoch0.json", {})
        vals = [tok.get(g, {}).get("mean_l2", np.nan) for g in groups]
        ax.bar(np.arange(len(groups)) + i * width - 0.4 + width / 2, vals,
               width=width, color=COLORS[k], label=NAMES[k])
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
    ax.set_ylabel("mean L2 norm at encoder input"); ax.grid(alpha=.3, axis="y")
    ax.legend()
    save(fig, "token_l2_norm.png")

    # 4) attention: score absmax / max weight / entropy over steps
    for field, ylabel, fname, logy in [
            ("score_absmax", "L0 |score| max", "attention_score_absmax.png", True),
            ("attn_max", "L0 max attention weight", "attention_max_weight.png", False),
            ("attn_entropy", "L0 attention entropy", "attention_entropy.png", False)]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in MODELS:
            p = run_dir(k) / "attention_diagnostics.jsonl"
            if not p.exists():
                continue
            xs, ys = [], []
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                L0 = next((x for x in r.get("layers", [])
                           if x.get("layer") == "L0" and field in x), None)
                if L0:
                    xs.append(r["step"]); ys.append(L0[field])
            if xs:
                ax.plot(xs, ys, marker=".", color=COLORS[k], label=NAMES[k])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("optimizer step"); ax.set_ylabel(ylabel)
        ax.grid(alpha=.3); ax.legend()
        save(fig, fname)

    # 5) residual stream std by layer (last diagnostic snapshot)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in MODELS:
        p = run_dir(k) / "attention_diagnostics.jsonl"
        if not p.exists():
            continue
        last = None
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "layers" in r:
                last = r
        if last is None:
            continue
        layers = sorted([x for x in last["layers"] if "residual_std" in x],
                        key=lambda x: int(x["layer"].split("L")[-1].split(".")[0]))
        ax.plot([int(x["layer"].split("L")[-1].split(".")[0]) for x in layers],
                [x["residual_std"] for x in layers], marker="o",
                color=COLORS[k], label=NAMES[k])
    ax.set_yscale("log"); ax.set_xlabel("encoder layer")
    ax.set_ylabel("residual stream std"); ax.grid(alpha=.3); ax.legend()
    save(fig, "residual_stream_std.png")
    return made


def to_markdown(summaries: dict[str, dict]) -> str:
    lines = ["# Stage 14-R2 稳定性验证汇总", ""]
    for k in MODELS:
        s = summaries[k]
        g = s["grad_norm"]
        lines += [f"## {s['model']}（{k}）", "",
                  f"- 参数量：{s['params']:,}" if s["params"] else "- 参数量：?",
                  f"- 完成 epoch 数：{s['epochs_run']}",
                  f"- 更新步数：{g['n_updates']}，跳过：{g['skipped_updates']}",
                  f"- grad norm 前 500 步中位数：{g['first500_median']:.3e}",
                  f"- grad norm 后 500 步中位数：{g['last500_median']:.3e}",
                  f"- grad norm 最大值：{g['max']:.3e}" if g["max"] else "- grad norm 最大值：?",
                  ""]
        if s["epochs"]:
            lines += ["| epoch | train loss | train acc | Dev Macro-F1 | Dev acc | 预测 S/F/L/R | 跳过 | 秒 |",
                      "|---:|---:|---:|---:|---:|---|---:|---:|"]
            for e in s["epochs"]:
                lines.append(
                    f"| {e['epoch']} | {e['train_loss']:.4f} | {e['train_acc']:.4f} | "
                    f"{e['dev_macro_f1']:.4f} | {e['dev_acc']:.4f} | "
                    f"{'/'.join(map(str, e['pred']))} | {e['skipped_updates']} | {e['seconds']:.0f} |")
            lines.append("")
        if s["aborted"]:
            lines += [f"**ABORTED**：{s['aborted'].get('reason')}", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    summaries = {k: summarize(k) for k in MODELS}
    (ROOT / "stability_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    figs = plot_all(summaries)
    # attention_diagnostics.json: merge snapshots for convenience
    merged = {}
    for k in MODELS:
        p = run_dir(k) / "attention_diagnostics.jsonl"
        recs = []
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        merged[NAMES[k]] = {"n_snapshots": len(recs),
                            "snapshots": recs}
    (ROOT / "attention_diagnostics.json").write_text(
        json.dumps(merged, indent=1) + "\n", encoding="utf-8")
    md = to_markdown(summaries)
    (ROOT / "stability_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print("figures:", figs)


if __name__ == "__main__":
    main()

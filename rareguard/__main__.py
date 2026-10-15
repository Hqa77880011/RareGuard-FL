import argparse
import json
from pathlib import Path

import numpy as np

from .data import prepare
from .run import METHODS, run


def plot(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    experiments = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    methods = sorted({r["config"]["method"] for r in experiments})
    for metric in ("rare_auprc", "macro_f1", "attack_success_rate", "worst_client_error", "ece"):
        means, deviations = [], []
        for method in methods:
            values = [r["test"][metric] for r in experiments if r["config"]["method"] == method
                      and r["test"][metric] is not None]
            means.append(float(np.mean(values)) if values else np.nan)
            deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(methods, means, yerr=deviations, color="#518393")
        ax.set_ylabel(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(destination / f"{metric}.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for path, experiment in zip(paths, experiments):
        history = json.loads((Path(path).parent / "history.json").read_text(encoding="utf-8"))
        ax.plot([r["communication_bytes"] / 1e6 for r in history],
                [r["validation"]["rare_auprc"] for r in history],
                label=f"{experiment['config']['method']} seed={experiment['config']['seed']}")
    ax.set(xlabel="Parameter communication (MB)", ylabel="Validation rare AUPRC")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "communication.png", dpi=180)
    plt.close(fig)
    summary = []
    for method in methods:
        for fraction in sorted({r["config"]["byzantine"] for r in experiments}):
            runs = [r for r in experiments if r["config"]["method"] == method
                    and r["config"]["byzantine"] == fraction]
            if runs:
                summary.append(dict(method=method, byzantine=fraction,
                                    mean={metric: float(np.mean([r["test"][metric] for r in runs
                                          if r["test"][metric] is not None]))
                                          for metric in runs[0]["test"]
                                          if any(r["test"][metric] is not None for r in runs)}))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(prog="python -m rareguard")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--manifest", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--window", type=int, default=4096)
    data.add_argument("--stride", type=int, default=2048)
    data.add_argument("--representation", choices=("waveform", "spectrum"), default="waveform")
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--rare-labels", type=int, nargs="+", required=True)
    train.add_argument("--method", choices=METHODS, default="rareguard")
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--rounds", type=int, default=100)
    train.add_argument("--local-epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--local-lr", type=float, default=0.01)
    train.add_argument("--server-lr", type=float, default=1)
    train.add_argument("--participation", type=float, default=0.4)
    train.add_argument("--byzantine", type=float, default=0.2)
    train.add_argument("--attack", choices=("sign_flip", "model_replacement", "label_flip", "mimicry"), default="mimicry")
    train.add_argument("--attack-target", type=int, default=0)
    train.add_argument("--attack-scale", type=float, default=10)
    train.add_argument("--mimic-strength", type=float, default=0.5)
    train.add_argument("--clip-multiplier", type=float, default=2)
    train.add_argument("--trim", type=float, default=0.2)
    train.add_argument("--projection", type=int, default=32)
    train.add_argument("--bootstraps", type=int, default=8)
    train.add_argument("--damping", type=float, default=10)
    train.add_argument("--cg-iterations", type=int, default=30)
    train.add_argument("--eta", type=float, default=1)
    train.add_argument("--kappa", type=float, default=1)
    train.add_argument("--influence-clip", type=float, default=1)
    train.add_argument("--utility-temperature", type=float, default=4)
    train.add_argument("--distance-temperature", type=float, default=1)
    train.add_argument("--distance-center", type=float, default=2)
    train.add_argument("--integrity-veto", type=float, default=6)
    train.add_argument("--maximum-weight", type=float, default=0.35)
    train.add_argument("--tail-fraction", type=float, default=0.2)
    train.add_argument("--tail-weight", type=float, default=0.5)
    train.add_argument("--server-steps", type=int, default=1)
    train.add_argument("--server-refine-lr", type=float, default=0.001)
    train.add_argument("--server-proximal", type=float, default=1)
    train.add_argument("--proximal", type=float, default=0.01)
    visual = commands.add_parser("plot")
    visual.add_argument("--metrics", nargs="+", required=True)
    visual.add_argument("--output", default="outputs/figures")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output, args.window, args.stride, args.representation)
    elif args.command == "train":
        run(args)
    else:
        plot(args.metrics, args.output)


if __name__ == "__main__":
    main()

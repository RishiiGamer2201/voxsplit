"""Consolidate VoxSplit results into report plots and a summary (Phase 7).

Reads the committed CSV logs and produces:
  - experiments/plots/*.png  (quality vs speaker count, robustness, count
    accuracy, training-length ablation)
  - experiments/RESULTS.md   (consolidated tables)

Run:  python experiments/make_report.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)


def read_results():
    rows = []
    with open(HERE / "eval_set_results.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["num_speakers"] = int(r["num_speakers"])
            for k in ("mean_si_sdri", "mean_pesq", "mean_stoi"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def by_tag(rows, tag):
    return {r["num_speakers"]: r for r in rows if r["tag"] == tag}


def plot_quality_vs_count(rows):
    """SI-SDRi vs speaker count: fixed-N best, recursive oracle, blind."""
    recur = by_tag(rows, "orpit_20k_recursive_oracle")
    blind = by_tag(rows, "orpit_blind_ml")
    # Best fixed-N per level (pretrained/finetuned dedicated models).
    fixedN = {2: by_tag(rows, "orpit_20k_2spk").get(2),
              3: by_tag(rows, "sepformer-libri3mix").get(3),
              4: by_tag(rows, "pit4_8k").get(4),
              5: by_tag(rows, "pit5_8k").get(5)}

    ks = [2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [fixedN[k]["mean_si_sdri"] for k in ks], "o-",
            label="Best fixed-N (dedicated model)")
    ax.plot(ks, [(by_tag(rows, "orpit_20k_2spk").get(2) if k == 2
                  else recur.get(k))["mean_si_sdri"] for k in ks], "s--",
            label="OR-PIT recursion (oracle count)")
    ax.plot(ks, [blind[k]["mean_si_sdri"] for k in ks], "^:",
            label="OR-PIT blind (unknown count)")
    ax.set_xlabel("Number of speakers")
    ax.set_ylabel("SI-SDRi (dB)")
    ax.set_title("Separation quality vs speaker count")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "quality_vs_count.png", dpi=130)
    plt.close(fig)


def plot_robustness(rows):
    """Clean/noise/reverb: clean-trained vs augmented-robust model."""
    clean_model = {"clean": by_tag(rows, "orpit_20k_2spk")[2]["mean_si_sdri"],
                   "noise": by_tag(rows, "orpit20k_noise")[2]["mean_si_sdri"],
                   "reverb": by_tag(rows, "orpit20k_reverb")[2]["mean_si_sdri"]}
    robust = {"clean": by_tag(rows, "orpit_robust_clean")[2]["mean_si_sdri"],
              "noise": by_tag(rows, "orpit_robust_noise")[2]["mean_si_sdri"],
              "reverb": by_tag(rows, "orpit_robust_reverb")[2]["mean_si_sdri"]}
    conds = ["clean", "noise", "reverb"]
    x = range(len(conds))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar([i - 0.2 for i in x], [clean_model[c] for c in conds], 0.4,
           label="Clean-trained OR-PIT")
    ax.bar([i + 0.2 for i in x], [robust[c] for c in conds], 0.4,
           label="Noise/reverb-augmented")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conds)
    ax.set_ylabel("SI-SDRi (dB)")
    ax.set_title("Robustness under degradation (2 speakers)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "robustness.png", dpi=130)
    plt.close(fig)


def blind_count_accuracy(tag="orpit20k_clfml8k"):
    acc = defaultdict(lambda: [0, 0])
    path = HERE / "phase4_count_predictions.csv"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["tag"] != tag:
                continue
            tk = int(r["true_k"])
            acc[tk][0] += int(int(r["pred_k"]) == tk)
            acc[tk][1] += 1
    return {k: v[0] / v[1] for k, v in sorted(acc.items()) if v[1]}


def overall_count_accuracy(tag):
    ok = tot = 0
    path = HERE / "phase4_count_predictions.csv"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["tag"] != tag:
                continue
            ok += int(int(r["pred_k"]) == int(r["true_k"]))
            tot += 1
    return ok / tot if tot else None


def plot_count_accuracy():
    acc = blind_count_accuracy()
    if not acc:
        return
    ks = list(acc.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(k) for k in ks], [acc[k] for k in ks], color="C2")
    ax.set_ylim(0, 1)
    ax.set_xlabel("True speaker count")
    ax.set_ylabel("Blind count accuracy")
    ax.set_title("Unknown-count accuracy (blind recursion)")
    for k in ks:
        ax.text(str(k), acc[k] + 0.02, f"{acc[k]:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(PLOTS / "count_accuracy.png", dpi=130)
    plt.close(fig)


def plot_training_length(rows):
    """6k vs 20k OR-PIT recursion at each level (training-length ablation)."""
    r6 = by_tag(rows, "orpit_6k_recursive_oracle")
    r20 = by_tag(rows, "orpit_20k_recursive_oracle")
    ks = [3, 4, 5]
    x = range(len(ks))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], [r6[k]["mean_si_sdri"] for k in ks], 0.4,
           label="6k steps")
    ax.bar([i + 0.2 for i in x], [r20[k]["mean_si_sdri"] for k in ks], 0.4,
           label="20k steps")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{k} spk" for k in ks])
    ax.set_ylabel("SI-SDRi (dB)")
    ax.set_title("Training-length ablation (recursive OR-PIT)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "training_length.png", dpi=130)
    plt.close(fig)


def write_results_md(rows):
    recur = by_tag(rows, "orpit_20k_recursive_oracle")
    blind = by_tag(rows, "orpit_blind_ml")
    acc = blind_count_accuracy()
    lines = [
        "# VoxSplit results summary (Phase 7)",
        "",
        "Frozen eval set: LibriSpeech test-clean, 20 mixtures per level, "
        "scored at 8 kHz. Metric: SI-SDRi (dB), higher is better.",
        "",
        "## Separation quality vs speaker count",
        "",
        "| Level | Best fixed-N | OR-PIT recursion (oracle count) | "
        "OR-PIT blind (unknown count) |",
        "|---|---|---|---|",
    ]
    fixedN = {2: ("OR-PIT 2-head", by_tag(rows, "orpit_20k_2spk")[2]),
              3: ("pretrained libri3mix", by_tag(rows, "sepformer-libri3mix")[3]),
              4: ("uPIT-4", by_tag(rows, "pit4_8k")[4]),
              5: ("uPIT-5", by_tag(rows, "pit5_8k")[5])}
    for k in (2, 3, 4, 5):
        name, fn = fixedN[k]
        rec = (by_tag(rows, "orpit_20k_2spk")[2] if k == 2 else recur[k])
        bl = blind.get(k)
        lines.append(
            f"| {k} spk | {fn['mean_si_sdri']:.2f} ({name}) | "
            f"{rec['mean_si_sdri']:.2f} | {bl['mean_si_sdri']:.2f} |")

    lines += ["", "## Blind unknown-count accuracy", "",
              "| Level | Count accuracy |", "|---|---|"]
    for k, a in acc.items():
        lines.append(f"| {k} spk | {a:.2f} |")

    lines += ["", "## Robustness (2 speakers)", "",
              "| Condition | Clean-trained | Augmented-robust |",
              "|---|---|---|"]
    for cond, clean_tag, rob_tag in [
            ("clean", "orpit_20k_2spk", "orpit_robust_clean"),
            ("WHAM noise", "orpit20k_noise", "orpit_robust_noise"),
            ("reverb", "orpit20k_reverb", "orpit_robust_reverb")]:
        c = by_tag(rows, clean_tag)[2]["mean_si_sdri"]
        rb = by_tag(rows, rob_tag)[2]["mean_si_sdri"]
        lines.append(f"| {cond} | {c:.2f} | {rb:.2f} |")

    # Ablations.
    lines += ["", "## Ablations", "",
              "**Fine-tuning (2-speaker, OR-PIT track).** Warm-start beats "
              "pretrained; more steps help until convergence.", "",
              "| Model | SI-SDRi |", "|---|---|"]
    for tag, name in [("sepformer-wsj02mix", "pretrained wsj02mix"),
                      ("orpit-6k", "OR-PIT 6k"),
                      ("orpit_20k_2spk", "OR-PIT 20k")]:
        r = by_tag(rows, tag).get(2)
        if r:
            lines.append(f"| {name} | {r['mean_si_sdri']:.2f} |")
    lines += ["",
              "**In-domain fine-tune does NOT help (3-speaker).** libri3mix is "
              "already LibriSpeech-trained.", "",
              "| Model | SI-SDRi |", "|---|---|",
              f"| pretrained libri3mix | "
              f"{by_tag(rows,'sepformer-libri3mix')[3]['mean_si_sdri']:.2f} |",
              f"| uPIT fine-tune 6k | "
              f"{by_tag(rows,'pit_libri3mix_6k')[3]['mean_si_sdri']:.2f} |",
              ""]

    clf_abl = [("orpit20k_clf8k", "clean-trained"),
               ("orpit20k_clfres8k", "residual (single-pass)"),
               ("orpit20k_clfml8k", "residual (multi-level)")]
    clf_rows = [(n, overall_count_accuracy(t)) for t, n in clf_abl]
    clf_rows = [(n, a) for n, a in clf_rows if a is not None]
    if clf_rows:
        lines += ["**Count/stop classifier training domain** (overall blind "
                  "count accuracy). Training on the separator's own residuals "
                  "at multiple depths is what makes blind stopping work.", "",
                  "| Classifier training | Count accuracy |", "|---|---|"]
        for n, a in clf_rows:
            lines.append(f"| {n} | {a:.2f} |")
        lines.append("")

    # Failure analysis.
    lines += ["## Failure analysis", "",
              "- **4 speakers is the count weak point** (0.40 blind accuracy): "
              "three recursion passes accumulate artifacts, so the residual "
              "count is hardest to read. Separation is fine once counted right "
              f"({by_tag(rows,'orpit_blind_ml')[4]['mean_si_sdri']:.2f} dB).",
              "- **More than 5 speakers (untrained regime).** A forced 6-speaker "
              "separation still runs but quality collapses to ~3.0 dB mean "
              "SI-SDRi (one source near-failed at -12.6 dB); the model was "
              "trained on 2-3 speaker mixtures and the field itself degrades "
              "hard past 5.",
              "- **Similar voices.** ECAPA discrimination weakens on short, "
              "bleed-heavy separated chunks, so same-gender similar-voice "
              "long-form conversations mis-stitch and mis-count, while "
              "distinct (e.g. mixed-gender) voices stitch cleanly.",
              "- **Reverberation** is the harshest degradation (clean model "
              f"{by_tag(rows,'orpit20k_reverb')[2]['mean_si_sdri']:.2f} dB); the "
              "augmented fine-tune recovers it only partially "
              f"({by_tag(rows,'orpit_robust_reverb')[2]['mean_si_sdri']:.2f} dB).",
              "- **Speech-enhancement post-filter hurts** (MetricGAN+ drops "
              f"clean SI-SDRi to "
              f"{by_tag(rows,'orpit20k_enhanced')[2]['mean_si_sdri']:.2f} dB); "
              "denoisers over-process already-separated speech.", ""]

    lines += ["## Plots", "",
              "![quality](plots/quality_vs_count.png)",
              "![robustness](plots/robustness.png)",
              "![count](plots/count_accuracy.png)",
              "![training](plots/training_length.png)", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = read_results()
    plot_quality_vs_count(rows)
    plot_robustness(rows)
    plot_count_accuracy()
    plot_training_length(rows)
    write_results_md(rows)
    print(f"Wrote plots to {PLOTS} and RESULTS.md")
    for p in sorted(PLOTS.glob("*.png")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

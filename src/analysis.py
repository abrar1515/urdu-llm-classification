#!/usr/bin/env python
"""Analysis for "Performance Comparison of LoRA-Fine-Tuned Large Language Models
for Urdu News Topic Classification".

Loads the files produced by the Kaggle notebook (no news text is needed), then writes
tables (CSV), summary.json and figures (PNG + PDF, 600 dpi by default).

Required in --results-dir:
    results.json (or result.json)       metrics, per-class reports, training history, tokenizer fertility
    preds_<model>_lora_r16.npy          test predictions of the three LLMs (class ids 0-3)
    preds_TFIDF.npy                     test predictions of the TF-IDF + LinearSVC baseline
    test_labels.csv                     gold test labels, column "y" (class ids 0-3)
Optional:
    label_audit.csv                     manual label audit, columns: test_row, check
                                        (check = "label OK" | "label wrong" | "unclear");
                                        picked up automatically if it is in --results-dir

Run src/prepare_inputs.py first to create preds_TFIDF.npy, test_labels.csv and label_audit.csv.

Usage (from the repository root):
    python src/analysis.py                       # figures -> figures/, tables + summary.json -> outputs/
    python src/analysis.py --dpi 300 --formats png
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# ----------------------------------------------------------------------------- configuration
CLASSES = ["Business & Economics", "Entertainment", "Science & Technology", "Sports"]
SHORT = ["Business & Econ.", "Entertainment", "Sci & Tech", "Sports"]
TFIDF = "TF-IDF + LinearSVC"
LLMS = ["Qwen2.5-3B", "Llama-3.2-3B", "Gemma-2-2B"]
NAMES = [TFIDF] + LLMS
# display name -> (key in results.json, candidate prediction files)
# (Kaggle downloads may replace "." by "_" in file names, so both spellings are tried)
SYSTEMS = {
    TFIDF: ("TFIDF+LinearSVC", ["preds_TFIDF.npy", "preds_TFIDF_reproduced.npy"]),
    "Qwen2.5-3B": ("Qwen2.5-3B_lora_r16", ["preds_Qwen2.5-3B_lora_r16.npy", "preds_Qwen2_5-3B_lora_r16.npy"]),
    "Llama-3.2-3B": ("Llama-3.2-3B_lora_r16", ["preds_Llama-3.2-3B_lora_r16.npy", "preds_Llama-3_2-3B_lora_r16.npy"]),
    "Gemma-2-2B": ("Gemma-2-2B_lora_r16", ["preds_Gemma-2-2B_lora_r16.npy", "preds_Gemma-2-2B_lora_r16.npy"]),
}
COLORS = {TFIDF: "#8c8c8c", "Qwen2.5-3B": "#d98c3f", "Llama-3.2-3B": "#3b7dd8", "Gemma-2-2B": "#3a9d6e"}


# ----------------------------------------------------------------------------- helpers
def find_file(folder, candidates):
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of {candidates} found in '{folder}'. "
                            f"Copy the Kaggle files into it and run src/prepare_inputs.py first.")


def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def point_metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=range(4))
    return accuracy_score(y, pred), macro_f1_from_cm(cm)


def ci95(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def save_figure(fig, fig_dir, name, dpi, formats):
    """Save one figure in every requested format (PNG is raster at `dpi`; PDF is vector)."""
    for fmt in formats:
        fig.savefig(os.path.join(fig_dir, f"{name}.{fmt}"), format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="outputs", help="tables/ and summary.json are written here")
    ap.add_argument("--figures-dir", default="figures", help="PNG and PDF figures are written here")
    ap.add_argument("--audit", default=None, help="CSV with columns test_row, check (default: <results-dir>/label_audit.csv if present)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--formats", default="png,pdf", help="comma-separated, e.g. png,pdf")
    ap.add_argument("--resamples", type=int, default=10_000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    formats = [f.strip().lower() for f in a.formats.split(",") if f.strip()]

    os.makedirs(os.path.join(a.out_dir, "tables"), exist_ok=True)
    os.makedirs(a.figures_dir, exist_ok=True)
    if a.audit is None and os.path.exists(os.path.join(a.results_dir, "label_audit.csv")):
        a.audit = os.path.join(a.results_dir, "label_audit.csv")
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts in PDFs (journal-friendly)
        "savefig.dpi": a.dpi,
    })

    # ---- load inputs (no article text needed)
    res = json.load(open(find_file(a.results_dir, ["results.json", "result.json"]), encoding="utf-8"))
    labels_path = find_file(a.results_dir, ["test_labels.csv", "test.csv"])
    y = pd.read_csv(labels_path)["y"].to_numpy().astype(int)
    preds = {k: np.load(find_file(a.results_dir, files)).astype(int) for k, (_, files) in SYSTEMS.items()}
    n = len(y)
    for k, p in preds.items():
        assert len(p) == n, f"{k}: {len(p)} predictions but {n} test labels"
    print(f"Loaded {n} test labels and predictions of {len(preds)} systems "
          f"(class counts: {np.bincount(y, minlength=4).tolist()})")
    for k, (key, _) in SYSTEMS.items():     # sanity check against the accuracy logged on Kaggle
        logged, now = res[key]["accuracy"], accuracy_score(y, preds[k])
        if abs(logged - now) > 1e-3:
            print(f"WARNING: {k}: accuracy {now:.4f} from predictions differs from {logged:.4f} in results.json")

    # ---- bootstrap (same resamples for every system -> paired)
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, n, size=(a.resamples, n))

    def bootstrap(pred):
        code = y * 4 + pred
        acc, f1 = np.empty(a.resamples), np.empty(a.resamples)
        for b in range(a.resamples):
            cm = np.bincount(code[idx[b]], minlength=16).reshape(4, 4)
            acc[b], f1[b] = np.trace(cm) / n, macro_f1_from_cm(cm)
        return acc, f1

    boot = {k: bootstrap(preds[k]) for k in NAMES}

    # ---- Table: main results
    rows = []
    for k in NAMES:
        acc, f1 = point_metrics(y, preds[k])
        p, r, _, _ = precision_recall_fscore_support(y, preds[k], average="macro", zero_division=0)
        fw = precision_recall_fscore_support(y, preds[k], average="weighted", zero_division=0)[2]
        (alo, ahi), (flo, fhi) = ci95(boot[k][0]), ci95(boot[k][1])
        rows.append(dict(model=k, accuracy=acc, acc_ci_low=alo, acc_ci_high=ahi, precision_macro=p, recall_macro=r,
                         f1_macro=f1, f1_ci_low=flo, f1_ci_high=fhi, f1_weighted=fw))
    main_df = pd.DataFrame(rows)
    main_df.to_csv(os.path.join(a.out_dir, "tables", "main_results.csv"), index=False)

    # ---- Table: pairwise exact McNemar + Holm + paired-bootstrap delta macro-F1
    pw = []
    for m1, m2 in combinations(NAMES, 2):
        c1, c2 = preds[m1] == y, preds[m2] == y
        only1, only2 = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
        p_val = binomtest(only1, only1 + only2, 0.5).pvalue if only1 + only2 else 1.0
        d = boot[m1][1] - boot[m2][1]
        pw.append(dict(model_a=m1, model_b=m2, only_a_correct=only1, only_b_correct=only2, mcnemar_p=p_val,
                       delta_f1=point_metrics(y, preds[m1])[1] - point_metrics(y, preds[m2])[1],
                       delta_ci_low=ci95(d)[0], delta_ci_high=ci95(d)[1]))
    pw = pd.DataFrame(pw)
    order, holm, running = np.argsort(pw.mcnemar_p.values), np.empty(len(pw)), 0.0
    for rank, i in enumerate(order):                    # Holm step-down adjustment
        running = max(running, min(1.0, pw.mcnemar_p.values[i] * (len(pw) - rank)))
        holm[i] = running
    pw["holm_p"] = holm
    pw.to_csv(os.path.join(a.out_dir, "tables", "pairwise_tests.csv"), index=False)

    # ---- Table: per-class precision / recall / F1
    pc = []
    for k in NAMES:
        p, r, f, s = precision_recall_fscore_support(y, preds[k], labels=range(4), zero_division=0)
        pc += [dict(model=k, cls=CLASSES[i], precision=p[i], recall=r[i], f1=f[i], support=int(s[i])) for i in range(4)]
    pc = pd.DataFrame(pc)
    pc.to_csv(os.path.join(a.out_dir, "tables", "per_class.csv"), index=False)

    # ---- Table: efficiency and tokenizer fertility (from results.json)
    fert = res["_token_fertility"]
    eff = pd.DataFrame([dict(
        model=k, trainable_params_M=res[SYSTEMS[k][0]]["trainable_params"] / 1e6,
        trainable_pct=res[SYSTEMS[k][0]]["trainable_pct"], train_time_min=res[SYSTEMS[k][0]]["train_time_s"] / 60,
        peak_vram_gb=res[SYSTEMS[k][0]]["peak_vram_gb"], inference_samples_per_s=res[SYSTEMS[k][0]]["test_samples_per_s"],
        fertility=fert[k], words_in_256_tokens=256 / fert[k]) for k in LLMS])
    eff.to_csv(os.path.join(a.out_dir, "tables", "efficiency.csv"), index=False)

    # ---- Error analysis (text-free)
    P3 = np.stack([preds[k] for k in LLMS])
    wrong_all = np.all(P3 != y, axis=0)
    majority = np.array([np.bincount(P3[:, j], minlength=4).argmax() for j in range(n)])
    pd.DataFrame({"test_row": np.where(wrong_all)[0], "gold": [CLASSES[i] for i in y[wrong_all]],
                  **{k: [CLASSES[i] for i in preds[k][wrong_all]] for k in LLMS}}) \
        .to_csv(os.path.join(a.out_dir, "tables", "shared_errors.csv"), index=False)
    summary = dict(
        n_test=n, bootstrap_resamples=a.resamples, seed=a.seed,
        errors_all_three_llms=int(wrong_all.sum()), share_all_three_wrong=float(wrong_all.mean()),
        at_least_one_llm_correct=float(1 - wrong_all.mean()),
        majority_vote_accuracy=float(accuracy_score(y, majority)), majority_vote_macro_f1=float(point_metrics(y, majority)[1]),
        pairwise_agreement={f"{m1} / {m2}": float(np.mean(preds[m1] == preds[m2])) for m1, m2 in combinations(LLMS, 2)},
        top_confusions={k: sorted([(int(confusion_matrix(y, preds[k], labels=range(4))[i, j]), SHORT[i], SHORT[j])
                                   for i in range(4) for j in range(4) if i != j
                                   and confusion_matrix(y, preds[k], labels=range(4))[i, j]], reverse=True)[:3] for k in NAMES})

    # ---- Optional: manual label audit and sensitivity analysis
    if a.audit:
        audit = pd.read_csv(a.audit)
        audit["check"] = audit["check"].astype(str).str.strip()
        bad_rows = set(audit.test_row) - set(np.where(wrong_all)[0])
        if bad_rows:
            print(f"WARNING: audit rows not among the shared errors: {sorted(bad_rows)}")
        counts = audit.check.value_counts().to_dict()
        wrong_rows = audit.loc[audit.check == "label wrong", "test_row"].to_numpy()
        unclear_rows = audit.loc[audit.check == "unclear", "test_row"].to_numpy()
        sens = []
        for label, drop in [("original", []), ("minus 'label wrong'", wrong_rows),
                            ("minus 'label wrong' and 'unclear'", np.concatenate([wrong_rows, unclear_rows]))]:
            keep = np.setdiff1d(np.arange(n), drop)
            for k in NAMES:
                acc, f1 = point_metrics(y[keep], preds[k][keep])
                sens.append(dict(test_set=label, n=len(keep), model=k, accuracy=acc, macro_f1=f1))
        pd.DataFrame(sens).to_csv(os.path.join(a.out_dir, "tables", "audit_sensitivity.csv"), index=False)
        summary["audit"] = dict(counts=counts, label_wrong_share_of_test=float(len(wrong_rows) / n))
        print("Audit counts:", counts, f"| label wrong = {len(wrong_rows)}/{n} = {len(wrong_rows) / n:.2%}")

    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2, ensure_ascii=False, default=float)

    # ================================================================= figures (PNG + PDF)
    # macro-F1 with bootstrap CI
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, k in enumerate(NAMES):
        r = main_df[main_df.model == k].iloc[0]
        ax.bar(i, r.f1_macro, color=COLORS[k], width=0.6, capsize=4, ecolor="black",
               yerr=[[r.f1_macro - r.f1_ci_low], [r.f1_ci_high - r.f1_macro]])
        ax.text(i, r.f1_ci_high + 0.003, f"{r.f1_macro:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels([k.replace(" + ", "\n+ ") for k in NAMES])
    ax.set_ylim(0.90, 0.985)
    ax.set_ylabel(f"Macro-F1 (test, n={n})")
    ax.set_title("Macro-F1 with 95% bootstrap CI (y-axis truncated at 0.90)", fontsize=10)
    save_figure(fig, a.figures_dir, "macro_f1_ci", a.dpi, formats)

    # confusion matrices (2 x 2)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for ax, k in zip(axes.ravel(), NAMES):
        cm = confusion_matrix(y, preds[k], labels=range(4))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(1).max())
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.55 * cm.sum(1).max() else "black")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(SHORT, rotation=35, ha="right", fontsize=8); ax.set_yticklabels(SHORT, fontsize=8)
        ax.set_title(f"{k} (acc {accuracy_score(y, preds[k]):.3f})", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.spines[:].set_visible(True)
    fig.tight_layout()
    save_figure(fig, a.figures_dir, "confusion_matrices", a.dpi, formats)

    # per-class F1
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    w = 0.2
    for i, k in enumerate(NAMES):
        ax.bar(np.arange(4) + (i - 1.5) * w, pc[pc.model == k].f1.values, w, color=COLORS[k], label=k)
    ax.set_xticks(range(4)); ax.set_xticklabels(SHORT)
    ax.set_ylim(0.85, 1.035); ax.set_ylabel("F1 (test)")
    ax.set_title("Per-class F1 (y-axis truncated at 0.85)", fontsize=10)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center")
    save_figure(fig, a.figures_dir, "per_class_f1", a.dpi, formats)

    # tokenizer fertility
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, k in enumerate(LLMS):
        r = eff[eff.model == k].iloc[0]
        ax.bar(i, r.fertility, color=COLORS[k], width=0.55)
        ax.text(i, r.fertility + 0.05, f"{r.fertility:.2f}\n(~{r.words_in_256_tokens:.0f} words\nin 256 tokens)",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(LLMS)
    ax.set_ylim(0, 4.4); ax.set_ylabel("Tokens per word (Urdu)")
    ax.set_title("Tokenizer fertility on Urdu training text", fontsize=10)
    save_figure(fig, a.figures_dir, "tokenizer_fertility", a.dpi, formats)

    # training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for k in LLMS:
        hist = res[SYSTEMS[k][0]]["history"]
        steps = [(e["step"], e["loss"]) for e in hist if "loss" in e]
        a1.plot(*zip(*steps), marker="o", ms=3, color=COLORS[k], label=k)
        evals = [(e["epoch"], e["eval_f1_macro"]) for e in hist if "eval_f1_macro" in e]
        a2.plot(*zip(*evals), marker="o", color=COLORS[k], label=k)
    a1.set_yscale("log"); a1.set_xlabel("Training step"); a1.set_ylabel("Training loss (log scale)")
    a1.set_title("Training loss", fontsize=10)
    a2.set_xlabel("Epoch"); a2.set_xticks([1, 2]); a2.set_ylabel("Validation macro-F1")
    a2.set_title("Validation macro-F1 per epoch", fontsize=10); a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, a.figures_dir, "training_curves", a.dpi, formats)

    print(main_df[["model", "accuracy", "f1_macro", "f1_ci_low", "f1_ci_high"]].round(4).to_string(index=False))
    print(pw.round(4).to_string(index=False))
    print(f"Wrote tables and summary.json to '{a.out_dir}' and figures ({', '.join(formats)}, {a.dpi} dpi) to '{a.figures_dir}'")


if __name__ == "__main__":
    main()
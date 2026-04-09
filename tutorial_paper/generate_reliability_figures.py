from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


PAPER_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAPER_DIR.parents[1]
SBI_DIR = ROOT_DIR / "sbi_paper_experiments"
PRIOR_DIR = SBI_DIR / "results" / "prior"
ENSEMBLE_DIR = SBI_DIR / "results" / "ensembles"
MISSPEC_DIR = SBI_DIR / "results" / "misspecification"
DATA_DIR = ROOT_DIR / "tutorials" / "data"

sys.path.insert(0, str(ROOT_DIR))
from sbi_paper_experiments.scripts.run_misspecification import (  # noqa: E402
    SIGMA,
    avecilla_at_chuong_gens,
    chuong_truncated,
)


sns.set_theme(style="whitegrid", context="talk")
torch.set_default_dtype(torch.float32)

PRIOR_COLORS = {"low": "#4878CF", "high": "#D65F5F"}
ENSEMBLE_COLORS = {
    "single": "#4878CF",
    "prolonged": "#6ACC65",
    "K3": "#D65F5F",
    "K5": "#B47CC7",
}
MISSPEC_COLORS = {"avecilla": "#D65F5F", "chuong": "#4878CF"}


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def sample_posterior(posterior, x_obs: torch.Tensor, n_samples: int) -> np.ndarray:
    kwargs = {"x": x_obs}
    try:
        samples = posterior.sample((n_samples,), show_progress_bars=False, **kwargs)
    except TypeError:
        samples = posterior.sample((n_samples,), **kwargs)
    return samples.detach().cpu().numpy()


def prior_sensitivity_figure(summary: dict) -> None:
    model_labels = {
        "avecilla": "Avecilla et al.",
        "zhou": "Zhou et al.",
        "vz": "Vande Zande et al.",
    }
    nsims = [10_000, 100_000]
    records = []
    for model, label in model_labels.items():
        for nsim in nsims:
            for prior in ["low", "high"]:
                path = PRIOR_DIR / f"posterior_{model}_npe_maf_prior{prior}_nsim{nsim}_ep200_summary.json"
                stats = load_json(path)
                records.append(
                    {
                        "model": model,
                        "model_label": label,
                        "nsim": nsim,
                        "condition": f"{label}\n{nsim // 1000}k",
                        "prior": prior,
                        "nmae": stats["nmae"],
                        "norm_interval_width": stats["norm_interval_width"],
                        "sbcc_abs_err": stats["sbcc_abs_err"],
                        "pred_rmse": stats["pred_rmse"],
                    }
                )
    df = pd.DataFrame(records)
    cond_order = [f"{model_labels[m]}\n{nsim // 1000}k" for m in model_labels for nsim in nsims]
    metrics = [
        ("nmae", "NMAE"),
        ("norm_interval_width", "Norm. interval width"),
        ("sbcc_abs_err", "SBCC abs. error"),
        ("pred_rmse", "Predictive RMSE"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(21.2, 5.8), constrained_layout=True)
    x = np.arange(len(cond_order))

    for ax, (metric, title) in zip(axes, metrics):
        for i, condition in enumerate(cond_order):
            subset = df[df["condition"] == condition]
            low = subset.loc[subset["prior"] == "low", metric].iloc[0]
            high = subset.loc[subset["prior"] == "high", metric].iloc[0]
            ax.plot([i - 0.12, i + 0.12], [low, high], color="#999999", lw=1.5, zorder=1)
            ax.scatter(i - 0.12, low, color=PRIOR_COLORS["low"], s=65, zorder=3)
            ax.scatter(i + 0.12, high, color=PRIOR_COLORS["high"], s=65, zorder=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_order, fontsize=8.5, rotation=24, ha="right")
        ax.tick_params(axis="x", pad=5)
        ax.margins(x=0.06)
        ax.set_ylabel("")

    axes[0].set_ylabel("Metric value")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PRIOR_COLORS["low"], markersize=9, label="Lower prior"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PRIOR_COLORS["high"], markersize=9, label="Higher prior"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=11, loc="upper right")
    fig.suptitle("Prior sensitivity across three SBI benchmarks", y=1.05, fontsize=16)
    fig.savefig(PAPER_DIR / "fig_prior_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    prior_summary = {}
    for model, label in model_labels.items():
        prior_summary[label] = {}
        for nsim in nsims:
            subset = df[(df["model"] == model) & (df["nsim"] == nsim)].set_index("prior")
            prior_summary[label][str(nsim)] = {
                metric: {
                    "low": float(subset.loc["low", metric]),
                    "high": float(subset.loc["high", metric]),
                }
                for metric, _ in metrics
            }
    summary["prior"] = prior_summary


def ensemble_figure(summary: dict) -> None:
    model_order = ["avecilla", "phylo", "vz"]
    model_labels = {"avecilla": "Avecilla et al.", "phylo": "Moshe et al.", "vz": "Vande Zande et al."}
    strategy_labels = {
        "single": "Single",
        "prolonged": "Prolonged",
        "K3": "K=3",
        "K5": "K=5",
    }
    records = []
    for model in model_order:
        meta = load_json(ENSEMBLE_DIR / f"{model}_ensemble_experiment.json")
        for result in meta["results"]:
            rtype = result["type"]
            if rtype == "single":
                strategy = "single"
            elif rtype == "prolonged":
                strategy = "prolonged"
            elif result.get("ensemble_size") == 3:
                strategy = "K3"
            elif result.get("ensemble_size") == 5:
                strategy = "K5"
            else:
                continue
            stats = load_json(ENSEMBLE_DIR / f"posterior_{result['tag']}_summary.json")
            records.append(
                {
                    "model": model_labels[model],
                    "strategy": strategy,
                    "nmae": stats["nmae"],
                    "sbcc_abs_err": stats["sbcc_abs_err"],
                    "norm_interval_width": stats["norm_interval_width"],
                    "wall_min": result["wall_s"] / 60.0,
                }
            )
    df = pd.DataFrame(records)
    metrics = [
        ("nmae", "NMAE"),
        ("sbcc_abs_err", "SBCC abs. error"),
        ("norm_interval_width", "Norm. interval width"),
        ("wall_min", "Wall time (min)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20.8, 5.8), constrained_layout=True)
    base_x = np.arange(len(model_order))
    width = 0.18
    strategy_order = ["single", "prolonged", "K3", "K5"]

    for ax, (metric, title) in zip(axes, metrics):
        for idx, strategy in enumerate(strategy_order):
            vals = []
            for model in [model_labels[m] for m in model_order]:
                row = df[(df["model"] == model) & (df["strategy"] == strategy)].iloc[0]
                vals.append(row[metric])
            xpos = base_x + (idx - 1.5) * width
            ax.bar(xpos, vals, width=width, color=ENSEMBLE_COLORS[strategy], label=strategy_labels[strategy])
        ax.set_title(title)
        ax.set_xticks(base_x)
        ax.set_xticklabels([model_labels[m] for m in model_order], fontsize=9, rotation=12, ha="right")
        ax.tick_params(axis="x", pad=4)
        ax.margins(x=0.08)
        ax.set_ylabel("")

    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle("Budget-fair single vs prolonged vs ensemble comparisons", y=1.05, fontsize=16)
    fig.savefig(PAPER_DIR / "fig_ensemble_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ensemble_summary = {}
    for model in [model_labels[m] for m in model_order]:
        subset = df[df["model"] == model].set_index("strategy")
        ensemble_summary[model] = {
            strategy_labels[strategy]: {
                "nmae": float(subset.loc[strategy, "nmae"]),
                "sbcc_abs_err": float(subset.loc[strategy, "sbcc_abs_err"]),
                "norm_interval_width": float(subset.loc[strategy, "norm_interval_width"]),
                "wall_min": float(subset.loc[strategy, "wall_min"]),
            }
            for strategy in strategy_order
        }
    summary["ensemble"] = ensemble_summary


def misspecification_figure(summary: dict) -> None:
    exp_meta = load_json(MISSPEC_DIR / "misspecification_experiment.json")
    trunc_gens = np.array(exp_meta["trunc_gens"], dtype=int)
    n_samples = 5000
    n_ppc = 40

    posterior_paths = {
        model["label"]: MISSPEC_DIR / Path(model["posterior_path"]).name
        for model in exp_meta["models"]
    }
    with posterior_paths["avecilla_misspec"].open("rb") as handle:
        post_ave = pickle.load(handle)
    with posterior_paths["chuong_correct"].open("rb") as handle:
        post_chu = pickle.load(handle)

    obs_df = pd.read_csv(DATA_DIR / "ltr.csv")
    replicate_names = obs_df.iloc[:, 0].tolist()
    full_gens = np.array([int(col) for col in obs_df.columns[1:]], dtype=int)
    obs_trunc = obs_df[[str(g) for g in trunc_gens]].to_numpy(float)
    obs_full = obs_df[[str(g) for g in full_gens]].to_numpy(float)
    trunc_boundary = int(trunc_gens[-1])

    delta_s = []
    delta_d = []
    trunc_rmse_ave = []
    trunc_rmse_chu = []
    full_rmse_ave = []
    full_rmse_chu = []
    rng = np.random.default_rng(123)
    predictive_store = []

    for i in range(len(replicate_names)):
        x_obs = torch.tensor(obs_trunc[i], dtype=torch.float32)
        samples_ave = sample_posterior(post_ave, x_obs, n_samples)
        samples_chu = sample_posterior(post_chu, x_obs, n_samples)

        delta_s.append(float(samples_ave[:, 0].mean() - samples_chu[:, 0].mean()))
        delta_d.append(float(samples_ave[:, 1].mean() - samples_chu[:, 1].mean()))

        idx_ave = rng.choice(len(samples_ave), n_ppc, replace=False)
        idx_chu = rng.choice(len(samples_chu), n_ppc, replace=False)

        trunc_preds_ave = []
        trunc_preds_chu = []
        full_preds_ave = []
        full_preds_chu = []

        for idx in idx_ave:
            trunc_preds_ave.append(
                np.clip(
                    avecilla_at_chuong_gens(samples_ave[idx], generations=trunc_gens)
                    + rng.normal(0, SIGMA, len(trunc_gens)),
                    0,
                    1,
                )
            )
            full_preds_ave.append(
                np.clip(
                    avecilla_at_chuong_gens(samples_ave[idx], generations=full_gens)
                    + rng.normal(0, SIGMA, len(full_gens)),
                    0,
                    1,
                )
            )

        for idx in idx_chu:
            trunc_preds_chu.append(
                np.clip(
                    chuong_truncated(samples_chu[idx], generations=trunc_gens)
                    + rng.normal(0, SIGMA, len(trunc_gens)),
                    0,
                    1,
                )
            )
            full_preds_chu.append(
                np.clip(
                    chuong_truncated(samples_chu[idx], generations=full_gens)
                    + rng.normal(0, SIGMA, len(full_gens)),
                    0,
                    1,
                )
            )

        trunc_preds_ave = np.asarray(trunc_preds_ave)
        trunc_preds_chu = np.asarray(trunc_preds_chu)
        full_preds_ave = np.asarray(full_preds_ave)
        full_preds_chu = np.asarray(full_preds_chu)

        mean_trunc_ave = np.mean(trunc_preds_ave, axis=0)
        mean_trunc_chu = np.mean(trunc_preds_chu, axis=0)
        mean_full_ave = np.mean(full_preds_ave, axis=0)
        mean_full_chu = np.mean(full_preds_chu, axis=0)

        trunc_rmse_ave.append(float(np.sqrt(np.mean((mean_trunc_ave - obs_trunc[i]) ** 2))))
        trunc_rmse_chu.append(float(np.sqrt(np.mean((mean_trunc_chu - obs_trunc[i]) ** 2))))
        full_rmse_ave.append(float(np.sqrt(np.mean((mean_full_ave - obs_full[i]) ** 2))))
        full_rmse_chu.append(float(np.sqrt(np.mean((mean_full_chu - obs_full[i]) ** 2))))

        predictive_store.append(
            {
                "replicate": replicate_names[i].replace("gap1_", "").replace("_", " ").upper(),
                "obs_full": obs_full[i],
                "full_preds_ave": full_preds_ave,
                "full_preds_chu": full_preds_chu,
            }
        )

    fig, axes = plt.subplots(2, len(replicate_names), figsize=(2.7 * len(replicate_names), 6.4), sharex=True, sharey=True)

    legend_handles = [
        plt.Line2D([0], [0], color="black", marker="o", markersize=4, lw=1.4, label="Observed data"),
        plt.Line2D([0], [0], color=MISSPEC_COLORS["avecilla"], lw=2.2, label="Avecilla et al. posterior median"),
        plt.Line2D([0], [0], color=MISSPEC_COLORS["chuong"], lw=2.2, label="Chuong et al. posterior median"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#BDBDBD", alpha=0.12, edgecolor="none", label="Truncated fit window"),
    ]

    for i, entry in enumerate(predictive_store):
        rep_label = entry["replicate"]
        obs = entry["obs_full"]
        for row, model_key, color, row_label in [
            (0, "full_preds_ave", MISSPEC_COLORS["avecilla"], "Avecilla et al.\nmisspecified"),
            (1, "full_preds_chu", MISSPEC_COLORS["chuong"], "Chuong et al.\ncorrect"),
        ]:
            ax = axes[row, i]
            preds = entry[model_key]
            lo = np.quantile(preds, 0.1, axis=0)
            mid = np.quantile(preds, 0.5, axis=0)
            hi = np.quantile(preds, 0.9, axis=0)
            ax.axvspan(full_gens[0], trunc_boundary, color="#BDBDBD", alpha=0.12, zorder=0)
            ax.fill_between(full_gens, lo, hi, color=color, alpha=0.18, zorder=1)
            ax.plot(full_gens, mid, color=color, lw=2.1, zorder=2)
            ax.plot(full_gens, obs, "o-", color="black", ms=3.8, lw=1.4, zorder=3)
            ax.axvline(trunc_boundary, color="#777777", ls="--", lw=1.0, alpha=0.8)
            ax.set_title(rep_label, fontsize=10)
            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks([int(full_gens[0]), trunc_boundary, int(full_gens[-1])])
            if i == 0:
                ax.set_ylabel(f"{row_label}\nCNV frequency")
            else:
                ax.set_ylabel("")
            if row == 1:
                ax.set_xlabel("Generation")

    fig.legend(handles=legend_handles, frameon=False, fontsize=9.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Misspecification becomes obvious only after forward prediction", y=1.06, fontsize=16)
    fig.savefig(PAPER_DIR / "fig_misspecification_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary["misspecification"] = {
        "delta_log_s_mean": float(np.mean(delta_s)),
        "delta_log_s_std": float(np.std(delta_s)),
        "delta_log_delta_mean": float(np.mean(delta_d)),
        "delta_log_delta_std": float(np.std(delta_d)),
        "truncated_rmse": {
            "avecilla": float(np.mean(trunc_rmse_ave)),
            "chuong": float(np.mean(trunc_rmse_chu)),
        },
        "full_rmse": {
            "avecilla": float(np.mean(full_rmse_ave)),
            "chuong": float(np.mean(full_rmse_chu)),
        },
    }


def main() -> None:
    summary: dict[str, object] = {}
    prior_sensitivity_figure(summary)
    ensemble_figure(summary)
    misspecification_figure(summary)
    with (PAPER_DIR / "reliability_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

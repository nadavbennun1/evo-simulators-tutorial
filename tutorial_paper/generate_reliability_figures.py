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
ENSEMBLE_DIR = ROOT_DIR / "Ensembles" / "prolonged" / "ensemble_studies"
MISSPEC_DIR = ROOT_DIR / "Ensembles" / "prolonged" / "misspec_studies"

sys.path.insert(0, str(ENSEMBLE_DIR))
from utils import load_tensor_from_csv  # noqa: E402


sns.set_theme(style="whitegrid", context="talk")
torch.set_default_dtype(torch.float32)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def maybe_mean_last_dim(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.float()
    if tensor.ndim > 1:
        tensor = tensor.mean(1)
    return tensor.cpu().numpy()


def calibration_ks(tensor: torch.Tensor) -> float:
    tensor = tensor.float()
    if tensor.ndim > 1:
        tensor = tensor.mean(1)
    xs, _ = torch.sort(tensor)
    expected = torch.linspace(1 / len(xs), 1, len(xs))
    return float(torch.max(torch.abs(xs - expected)).item())


def evaluate_posterior_fast(posterior, theta_test: torch.Tensor, x_test: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    accs = torch.empty(len(theta_test), len(theta_test[0]))
    conf = torch.empty(len(theta_test), len(theta_test[0]))
    unc = torch.empty(len(theta_test), len(theta_test[0]))
    logps = torch.empty(len(theta_test))
    for i in range(len(theta_test)):
        posterior = posterior.set_default_x(x_test[i])
        try:
            samples = posterior.sample((num_samples,), show_progress_bars=False)
        except TypeError:
            samples = posterior.sample((num_samples,))
        try:
            theta_lp = posterior.log_prob(theta_test[i].unsqueeze(0))
            sample_lp = posterior.log_prob(samples)
        except TypeError:
            theta_lp = posterior.log_prob(theta_test[i].unsqueeze(0), x=x_test[i])
            sample_lp = posterior.log_prob(samples, x=x_test[i])
        accs[i] = samples.mean(0) - theta_test[i]
        unc[i] = samples.max(0).values - samples.min(0).values
        conf[i] = (sample_lp > theta_lp).sum() / num_samples
        logps[i] = theta_lp.mean()
    return accs, unc, conf, float(logps.mean().item())


def prior_sensitivity_figure(summary: dict) -> None:
    base = ENSEMBLE_DIR / "chuong"
    theta_test = load_tensor_from_csv(base / "theta_test.csv")
    x_test = load_tensor_from_csv(base / "x_test.csv")

    center = theta_test.mean(0, keepdim=True)
    idx = int(torch.argmin(((theta_test - center) ** 2).sum(1)).item())
    theta_true = theta_test[idx]
    x_obs = x_test[idx]

    posterior_specs = {
        "Lower prior": {
            "posterior": load_pickle(base / "posterior_low_40k.pkl"),
            "range": np.array([[-2.0, -1.0], [-7.0, -3.0], [-7.0, -3.0]]),
            "color": "#b35806",
            "accs": base / "accs_npe_low_40k.pt",
            "unc": base / "unc_npe_low_40k.pt",
        },
        "Oracle prior": {
            "posterior": load_pickle(base / "posterior_oracle_40k.pkl"),
            "range": np.array([[-1.5, -1.0], [-6.0, -3.0], [-6.0, -3.0]]),
            "color": "#1b9e77",
            "accs": base / "accs_npe_oracle_40k.pt",
            "unc": base / "unc_npe_oracle_40k.pt",
        },
        "Wider prior": {
            "posterior": load_pickle(base / "posterior_high_40k.pkl"),
            "range": np.array([[-1.5, -0.5], [-6.0, -2.0], [-6.0, -2.0]]),
            "color": "#7570b3",
            "accs": base / "accs_npe_high_40k.pt",
            "unc": base / "unc_npe_high_40k.pt",
        },
    }

    samples = {}
    for name, spec in posterior_specs.items():
        samples[name] = spec["posterior"].set_default_x(x_obs).sample((4000,)).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), tight_layout=True)
    param_labels = [r"$\log_{10}s$", r"$\log_{10}\delta$", r"$\log_{10}\phi$"]
    xlims = [(-2.05, -0.45), (-7.2, -1.8), (-7.2, -1.8)]

    for j, ax in enumerate(axes):
        for name, spec in posterior_specs.items():
            lo, hi = spec["range"][j]
            ax.axvspan(lo, hi, color=spec["color"], alpha=0.08)
            sns.kdeplot(samples[name][:, j], ax=ax, color=spec["color"], lw=2, fill=False, label=name if j == 0 else None)
        ax.axvline(float(theta_true[j]), color="black", ls="--", lw=1.5)
        ax.set_title(param_labels[j])
        ax.set_xlim(*xlims[j])
        ax.set_ylabel("Density" if j == 0 else "")
        ax.set_xlabel("")
    axes[0].legend(frameon=False, fontsize=11, loc="upper left")
    fig.suptitle("Posterior estimates shift with the training prior", y=1.05, fontsize=16)
    fig.savefig(PAPER_DIR / "fig_prior_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    prior_stats = {}
    for name, spec in posterior_specs.items():
        accs = torch.load(spec["accs"]).float()
        unc = torch.load(spec["unc"]).float()
        prior_stats[name] = {
            "mean_abs_error": float(accs.abs().mean().item()),
            "mean_uncertainty": float(unc.mean().item()),
        }
    summary["prior"] = {
        "representative_index": idx,
        "true_theta": [float(x) for x in theta_true.tolist()],
        "stats": prior_stats,
    }


def ensemble_figure(summary: dict) -> None:
    bagging_base = ENSEMBLE_DIR / "phylo" / "ensemble" / "bagging"
    vanilla_base = ENSEMBLE_DIR / "phylo" / "ensemble" / "vanilla"

    ensemble_stats = {
        "Single NPE": {
            "mae": torch.load(bagging_base / "accs_npe_single.pt").abs().mean(1).cpu().numpy(),
            "unc": torch.load(bagging_base / "unc_npe_single.pt").mean(1).cpu().numpy(),
            "conf": maybe_mean_last_dim(torch.load(bagging_base / "conf_npe_single.pt")),
            "logp": float(torch.load(bagging_base / "lpm_npe_single.pt").item()),
            "ks": calibration_ks(torch.load(bagging_base / "conf_npe_single.pt")),
            "color": "#d95f02",
        },
        "Vanilla ensemble": {
            "mae": torch.load(vanilla_base / "accs_npe_vanilla.pt").abs().mean(1).cpu().numpy(),
            "unc": torch.load(vanilla_base / "unc_npe_vanilla.pt").mean(1).cpu().numpy(),
            "conf": maybe_mean_last_dim(torch.load(vanilla_base / "conf_npe_vanilla.pt")),
            "logp": float(torch.load(vanilla_base / "lpm_npe_vanilla.pt").item()),
            "ks": calibration_ks(torch.load(vanilla_base / "conf_npe_vanilla.pt")),
            "color": "#7570b3",
        },
        "Bagging ensemble": {
            "mae": torch.load(bagging_base / "accs_npe_bagging.pt").abs().mean(1).cpu().numpy(),
            "unc": torch.load(bagging_base / "unc_npe_bagging.pt").mean(1).cpu().numpy(),
            "conf": maybe_mean_last_dim(torch.load(bagging_base / "conf_npe_bagging.pt")),
            "logp": float(torch.load(bagging_base / "lpm_npe_bagging.pt").item()),
            "ks": calibration_ks(torch.load(bagging_base / "conf_npe_bagging.pt")),
            "color": "#d95f02",
        }
    }
    ensemble_stats["Bagging ensemble"]["color"] = "#1b9e77"
    ensemble_stats["Single NPE"]["color"] = "#d95f02"
    ensemble_stats["Vanilla ensemble"]["color"] = "#7570b3"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), tight_layout=True)
    titles = ["Mean absolute error", "Mean posterior width", "Calibration diagnostic"]
    xlabels = ["Per-observation error", "Per-observation uncertainty", "Posterior confidence rank"]

    for name, stats in ensemble_stats.items():
        color = stats["color"]
        for ax, key in zip(axes[:2], ["mae", "unc"]):
            x, y = ecdf(stats[key])
            ax.plot(x, y, color=color, lw=2, label=name if key == "mae" else None)
        x, y = ecdf(stats["conf"])
        axes[2].plot(x, y, color=color, lw=2, label=name)

    axes[2].plot([0, 1], [0, 1], color="black", ls="--", lw=1, alpha=0.6)
    for ax, title, xlabel in zip(axes, titles, xlabels):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ECDF")
    axes[0].legend(frameon=False, fontsize=11, loc="lower right")
    fig.suptitle("Bagging modestly stabilizes ensemble NPE on the phylogeny benchmark", y=1.05, fontsize=16)
    fig.savefig(PAPER_DIR / "fig_ensemble_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary["ensemble"] = {
        name: {
            "mean_mae": float(stats["mae"].mean()),
            "mean_uncertainty": float(stats["unc"].mean()),
            "mean_confidence": float(stats["conf"].mean()),
            "mean_log_prob": stats["logp"],
            "calibration_ks": stats["ks"],
        }
        for name, stats in ensemble_stats.items()
    }


def misspecification_figure(summary: dict) -> None:
    metric_files = {
        "Matched WF": MISSPEC_DIR / "test_sims" / "metrics_WF.csv",
        "WF + DFE mismatch": MISSPEC_DIR / "test_sims" / "metrics_WF_DFE.csv",
        "WF + bottleneck mismatch": MISSPEC_DIR / "test_sims" / "metrics_WF_bottleneck.csv",
    }
    dfs = []
    for label, path in metric_files.items():
        df = pd.read_csv(path)
        df = df.rename(columns={"Unnamed: 0": "architecture"})
        df["model_label"] = label
        dfs.append(df)
    metrics = pd.concat(dfs, ignore_index=True)

    arch_order = [a for a in ["nle", "nre", "npe", "npse", "abc", "mamba_de"] if a in set(metrics["architecture"])]
    model_order = list(metric_files.keys())
    colors = ["#1b9e77", "#d95f02", "#7570b3"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), tight_layout=True)
    width = 0.24
    x = np.arange(len(arch_order))

    for i, model in enumerate(model_order):
        subset = metrics[metrics["model_label"] == model].set_index("architecture").loc[arch_order]
        axes[0].bar(x + (i - 1) * width, subset["large either"].astype(float), width=width, color=colors[i], label=model)
        axes[1].bar(x + (i - 1) * width, subset["rmse"].astype(float), width=width, color=colors[i], label=model)

    axes[0].set_title("Frequency of large parameter error")
    axes[0].set_ylabel("Fraction of test cases")
    axes[1].set_title("Posterior-predictive RMSE")
    axes[1].set_ylabel("RMSE")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() if a != "mamba_de" else "MAMBA-DE" for a in arch_order], rotation=20)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle("Misspecified simulators degrade recovery and predictive fit", y=1.05, fontsize=16)
    fig.savefig(PAPER_DIR / "fig_misspecification_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary["misspecification"] = {
        model: {
            "mean_large_either": float(metrics.loc[metrics["model_label"] == model, "large either"].astype(float).mean()),
            "mean_rmse": float(metrics.loc[metrics["model_label"] == model, "rmse"].astype(float).mean()),
        }
        for model in model_order
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

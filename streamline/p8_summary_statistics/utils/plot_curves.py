from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme()


def plot_model_roc(
    full_path: str,
    algorithm: str,
    abbrev: str,
    color,
    cv_partitions: int,
    mean_fpr: np.ndarray,
    tprs: List[np.ndarray],
    aucs: List[float],
    alg_result_table: List[List[Any]],
    show_plots: bool = False,
):
    """
    Per-algorithm ROC plot across CV folds.
    """
    # Define values for the mean ROC line (mean of individual CVs)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.rcParams["figure.figsize"] = (6, 6)

    for i in range(cv_partitions):
        plt.plot(
            alg_result_table[i][0],
            alg_result_table[i][1],
            lw=1,
            alpha=0.3,
            label="ROC fold %d (AUC = %0.3f)" % (i, alg_result_table[i][2]),
        )

    # No-skill line
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="No-Skill", alpha=0.8)

    # Mean ROC
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color=color,
        label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (float(mean_auc), float(std_auc)),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(algorithm)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    out = Path(full_path) / "model_evaluation" / f"{abbrev}_ROC.png"
    plt.savefig(out.as_posix(), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close("all")


def plot_model_prc(
    full_path: str,
    algorithm: str,
    abbrev: str,
    color,
    cv_partitions: int,
    mean_recall: np.ndarray,
    precs: List[np.ndarray],
    praucs: List[float],
    alg_result_table: List[List[Any]],
    outcome_label: str,
    data_name: str,
    instance_label: str | None = None,
    rep_data: pd.DataFrame | None = None,
    replicate: bool = False,
    show_plots: bool = False,
):
    """
    Per-algorithm PRC plot across CV folds.
    """
    mean_prec = np.mean(precs, axis=0)
    mean_pr_auc = np.mean(praucs)

    plt.rcParams["figure.figsize"] = (6, 6)

    for i in range(cv_partitions):
        plt.plot(
            alg_result_table[i][4],
            alg_result_table[i][3],
            lw=1,
            alpha=0.3,
            label="PRC fold %d (AUC = %0.3f)" % (i, alg_result_table[i][5]),
        )

    # no-skill line
    if not replicate:
        test = pd.read_csv(
            Path(full_path) / "CVDatasets" / f"{data_name}_CV_0_Test.csv"
        )
        if instance_label is not None and instance_label in test.columns:
            test = test.drop(instance_label, axis=1)
        test_y = test[outcome_label].values
    else:
        test_y = rep_data[outcome_label].values

    no_skill = len(test_y[test_y == 1]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], color="black", linestyle="--", label="No-Skill", alpha=0.8)

    std_pr_auc = np.std(praucs)
    plt.plot(
        mean_recall,
        mean_prec,
        color=color,
        label=r"Mean PRC (AUC = %0.3f $\pm$ %0.3f)" % (float(mean_pr_auc), float(std_pr_auc)),
        lw=2,
        alpha=0.8,
    )

    std_prec = np.std(precs, axis=0)
    precs_upper = np.minimum(mean_prec + std_prec, 1)
    precs_lower = np.maximum(mean_prec - std_prec, 0)
    plt.fill_between(
        mean_recall,
        precs_lower,
        precs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title(algorithm)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    out = Path(full_path) / "model_evaluation" / f"{abbrev}_PRC.png"
    plt.savefig(out.as_posix(), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close("all")


def plot_summary_roc(
    full_path: str,
    colors: Dict[str, Any],
    result_table: pd.DataFrame,
    show_plots: bool = False,
):
    """
    Summary ROC over algorithms (already averaged across CVs).
    """
    for alg in result_table.index:
        plt.plot(
            result_table.loc[alg]["fpr"],
            result_table.loc[alg]["tpr"],
            color=colors[alg],
            label="{}, AUC={:.3f}".format(alg, result_table.loc[alg]["auc"]),
        )

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="No-Skill", alpha=0.8)
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    out = Path(full_path) / "model_evaluation" / "Summary_ROC.png"
    plt.savefig(out.as_posix(), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close("all")


def plot_summary_prc(
    full_path: str,
    colors: Dict[str, Any],
    result_table: pd.DataFrame,
    outcome_label: str,
    data_name: str,
    instance_label: str | None = None,
    rep_data: pd.DataFrame | None = None,
    replicate: bool = False,
    show_plots: bool = False,
):
    """
    Summary PRC over algorithms (already averaged across CVs).
    """
    for alg in result_table.index:
        plt.plot(
            result_table.loc[alg]["recall"],
            result_table.loc[alg]["prec"],
            color=colors[alg],
            label="{}, AUC={:.3f}, APS={:.3f}".format(
                alg, result_table.loc[alg]["pr_auc"], result_table.loc[alg]["ave_prec"]
            ),
        )

    if not replicate:
        test = pd.read_csv(
            Path(full_path) / "CVDatasets" / f"{data_name}_CV_0_Test.csv"
        )
        if instance_label is not None and instance_label in test.columns:
            test = test.drop(instance_label, axis=1)
        test_y = test[outcome_label].values
    else:
        test_y = rep_data[outcome_label].values

    no_skill = len(test_y[test_y == 1]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], color="black", linestyle="--", label="No-Skill", alpha=0.8)

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall (Sensitivity)", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision (PPV)", fontsize=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    out = Path(full_path) / "model_evaluation" / "Summary_PRC.png"
    plt.savefig(out.as_posix(), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close("all")


def plot_metric_boxplots(
    full_path: str,
    algorithms: List[str],
    metrics: List[str],
    metric_dict: Dict[str, Dict[str, List[float]]],
    show_plots: bool = False,
):
    """
    Export boxplots comparing algorithm performance for each metric.
    """
    out_dir = Path(full_path) / "model_evaluation" / "metricBoxplots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        temp_list = []
        for alg in algorithms:
            temp_list.append(metric_dict[alg][metric])

        td = pd.DataFrame(temp_list).transpose().astype("float")
        td.columns = algorithms

        ax = td.plot(kind="box", rot=90)
        ax.set_ylabel(str(metric))
        ax.set_xlabel("ML Algorithm")

        out = out_dir / f"Compare_{metric}.png"
        plt.savefig(out.as_posix(), bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close("all")


def plot_ensemble_roc_summary(
    ens_root: Path,
    roc_summary: Dict[str, Dict[str, Any]],
    show_plots: bool = False,
):
    if not roc_summary:
        return

    plt.figure(figsize=(6, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, ens_id in enumerate(sorted(roc_summary.keys())):
        c = color_cycle[idx % len(color_cycle)]
        d = roc_summary[ens_id]
        plt.plot(
            d["fpr"],
            d["tpr"],
            color=c,
            label=f"{ens_id}, AUC={d['auc']:.3f}",
        )

    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="No-Skill", alpha=0.8)
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="lower right", fontsize=8)
    plt.title("Ensemble ROC Summary")

    out_path = ens_root / "Summary_ROC_ensembles.png"
    plt.savefig(out_path.as_posix(), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close("all")


def plot_ensemble_prc_summary(
    ens_root: Path,
    prc_summary: Dict[str, Dict[str, Any]],
    full_path: str,
    data_name: str,
    outcome_label: str,
    instance_label: str | None = None,
    show_plots: bool = False,
):
    if not prc_summary:
        return

    # no-skill baseline from first test set
    try:
        test = pd.read_csv(
            Path(full_path) / "CVDatasets" / f"{data_name}_CV_0_Test.csv"
        )
        if instance_label is not None and instance_label in test.columns:
            test = test.drop(instance_label, axis=1)
        test_y = test[outcome_label].values
        no_skill = len(test_y[test_y == 1]) / len(test_y)
    except Exception:
        no_skill = 0.5

    plt.figure(figsize=(6, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, ens_id in enumerate(sorted(prc_summary.keys())):
        c = color_cycle[idx % len(color_cycle)]
        d = prc_summary[ens_id]
        plt.plot(
            d["recall"],
            d["precision"],
            color=c,
            label=f"{ens_id}, AUC={d['pr_auc']:.3f}, APS={d['aps']:.3f}",
        )

    plt.plot([0, 1], [no_skill, no_skill], color="black", linestyle="--", label="No-Skill", alpha=0.8)
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall (Sensitivity)", fontsize=15)
    plt.ylabel("Precision (PPV)", fontsize=15)
    plt.legend(loc="lower left", fontsize=8)
    plt.title("Ensemble PRC Summary")

    out_path = ens_root / "Summary_PRC_ensembles.png"
    plt.savefig(out_path.as_posix(), bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close("all")

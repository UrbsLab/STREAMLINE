from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st

from streamline.p11_reporting.reporting import ReportPhaseJob


def _load_or_build_summary(exp_root: Path, outcome_label: str, outcome_type: str) -> Dict[str, Any]:
    report_dir = exp_root / "reporting"
    report_dir.mkdir(exist_ok=True)
    json_path = report_dir / "report_data.json"
    if json_path.is_file():
        with json_path.open() as f:
            return json.load(f)
    # fallback: build summary in-place (no PDF)
    job = ReportPhaseJob(
        output_path=str(exp_root.parent),
        experiment_name=str(exp_root.name),
        outcome_label=outcome_label,
        outcome_type=outcome_type,
        make_pdf=False,
    )
    summary = job.collect_summary()
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, default=ReportPhaseJob._json_default)
    return summary


def main():
    st.set_page_config(layout="wide")
    st.title("STREAMLINE Experiment Report (Phase 10)")

    st.sidebar.header("Experiment selection")
    output_path = st.sidebar.text_input("Output path", value="out")
    experiment_name = st.sidebar.text_input("Experiment name", value="exp_integration")
    outcome_label = st.sidebar.text_input("Outcome label", value="Class")
    outcome_type = st.sidebar.selectbox("Outcome type", ["Binary", "Multiclass", "Continuous"], index=0)

    if not output_path or not experiment_name:
        st.info("Enter output_path and experiment_name on the sidebar.")
        return

    exp_root = Path(output_path) / experiment_name
    if not exp_root.is_dir():
        st.error(f"Experiment folder not found: {exp_root}")
        return

    summary = _load_or_build_summary(exp_root, outcome_label, outcome_type)
    st.sidebar.success("Summary loaded.")

    st.subheader(f"Experiment: {summary.get('experiment_name', experiment_name)}")

    tabs = st.tabs(["Datasets", "Cross-dataset stats", "Raw JSON"])

    # ----- Datasets tab -----
    with tabs[0]:
        ds_list = summary.get("datasets", [])
        if not ds_list:
            st.write("_No datasets found._")
        else:
            ds_names = [d["name"] for d in ds_list]
            ds_name = st.selectbox("Dataset", ds_names)
            ds = next(d for d in ds_list if d["name"] == ds_name)

            st.markdown(f"### Dataset: `{ds_name}`")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Base models – mean metrics**")
                base_mean = ds.get("base_metrics_mean")
                if base_mean:
                    df_mean = pd.DataFrame.from_dict(base_mean, orient="index")
                    st.dataframe(df_mean)
                else:
                    st.write("_No base model summary found._")

            with col2:
                st.markdown("**Ensembles – mean metrics**")
                ens_mean = ds.get("ensemble_metrics_mean")
                if ens_mean:
                    df_ens = pd.DataFrame.from_dict(ens_mean, orient="index")
                    st.dataframe(df_ens)
                else:
                    st.write("_No ensemble summary found._")

            st.markdown("**Plots**")
            plots = ds.get("plots", {})
            # base summary plots
            for key in ["summary_roc", "summary_prc"]:
                p = plots.get(key)
                if p:
                    st.image(str((exp_root / "reporting" / p).resolve()), caption=key)

            # feature importance plots
            for p in plots.get("feature_importance", []) or []:
                st.image(str((exp_root / "reporting" / p).resolve()), caption="Feature importance")

            ens_plots = plots.get("ensembles", {})
            for key, p in ens_plots.items():
                st.image(str((exp_root / "reporting" / p).resolve()), caption=key)

            st.markdown("**Runtimes**")
            rt = ds.get("runtimes")
            if rt:
                df_rt = pd.DataFrame.from_dict(rt, orient="index")
                st.dataframe(df_rt)
            else:
                st.write("_No runtimes.csv found for this dataset._")

    # ----- Cross-dataset stats tab -----
    with tabs[1]:
        comp = summary.get("dataset_comparisons", {})
        if not comp:
            st.write("_No DatasetComparisons folder found._")
        else:
            csvs = comp.get("csvs", {})
            st.markdown("### Comparison CSVs")
            for label, path in csvs.items():
                if path is None:
                    continue
                if isinstance(path, list):
                    for p in path:
                        st.write(f"- **{label}**: `{p}`")
                else:
                    st.write(f"- **{label}**: `{path}`")

            st.markdown("### Boxplot images")
            for p in comp.get("boxplots", []):
                st.image(str((exp_root / "reporting" / p).resolve()), caption=p)

    # ----- Raw JSON tab -----
    with tabs[2]:
        st.json(summary)


if __name__ == "__main__":
    main()

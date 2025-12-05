from __future__ import annotations

from pathlib import Path
from typing import Optional

from streamline.p11_reporting.reporting import ReportPhaseJob


class P10Runner:
    """
    Thin runner wrapper around ReportPhaseJob, mainly for API symmetry.
    """

    def __init__(
        self,
        output_path: str,
        experiment_name: str,
        outcome_label: str = "Class",
        outcome_type: str = "Binary",
        instance_label: Optional[str] = None,
        make_pdf: bool = True,
    ):
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.outcome_label = outcome_label
        self.outcome_type = outcome_type
        self.instance_label = instance_label
        self.make_pdf = make_pdf

        exp_root = Path(output_path) / experiment_name
        if not exp_root.is_dir():
            raise RuntimeError(f"Experiment folder not found: {exp_root}")

    def run(self):
        ReportPhaseJob(
            output_path=self.output_path,
            experiment_name=self.experiment_name,
            outcome_label=self.outcome_label,
            outcome_type=self.outcome_type,
            instance_label=self.instance_label,
            make_pdf=self.make_pdf,
        ).run()

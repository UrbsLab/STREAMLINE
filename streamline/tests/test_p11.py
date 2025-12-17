from pathlib import Path
import pytest

from streamline.p11_reporting.reporting import ReportPhaseJob

@pytest.mark.integration
def test_p11_reporting_smoke():
    # This test assumes DemoData has already been run through prior phases
    # and outputs exist under out/DemoRun. If your CI runs all phases first,
    # keep as-is; otherwise, wire it after your all-phases integration test.
    output_path = "out"
    experiment_name = "DemoRun"

    ReportPhaseJob(output_path=output_path, experiment_name=experiment_name).run()

    exp_root = Path(output_path) / experiment_name
    rep = exp_root / "reporting"
    assert (rep / "report.html").is_file()
    assert (rep / "report.pdf").is_file()
    assert (rep / "report_data.json").is_file()

    # sanity: figures folder exists (may be empty if plotly export unavailable)
    assert (rep / "figures").is_dir()

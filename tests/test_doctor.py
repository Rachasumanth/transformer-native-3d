import json
import tempfile
from pathlib import Path

from t3d.doctor import main, run_doctor


def test_doctor_reports_missing_checkpoints():
    with tempfile.TemporaryDirectory() as tmp:
        report = run_doctor(artifacts=Path(tmp), check_output=Path(tmp), verify=False)
        assert report['ok'] is False
        names = {check['name'] for check in report['checks']}
        assert {'python', 'dependency:torch', 'checkpoint_manifest', 'output_writable'} <= names
        assert report['warnings']


def test_doctor_json_exit_code():
    with tempfile.TemporaryDirectory() as tmp:
        code = main(['--artifacts', tmp, '--output', tmp, '--no-verify', '--json'])
        assert code == 1

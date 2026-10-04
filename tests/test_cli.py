from vame_motifs.cli import main


def test_validate_ok(experiment, capsys):
    assert main(["validate", "-c", str(experiment)]) == 0
    assert "2 file(s) valid" in capsys.readouterr().out


def test_user_error_exit_code_and_stderr(tmp_path, capsys):
    assert main(["validate", "-c", str(tmp_path / "missing.yaml")]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Error:") and "Traceback" not in captured.err


def test_status_before_anything_ran(experiment, capsys):
    assert main(["status", "-c", str(experiment)]) == 0
    assert "not created" in capsys.readouterr().out

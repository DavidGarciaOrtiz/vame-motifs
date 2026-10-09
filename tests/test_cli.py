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


def test_gif_options_are_parsed():
    from vame_motifs.cli import build_parser

    args = build_parser().parse_args(["gif", "-c", "x.yaml", "--session", "rat01", "--start", "10", "--label", "motif"])
    assert (args.session, args.start, args.length, args.label, args.subtract_background) == ("rat01", 10, 500, "motif", False)


def test_umap_without_segmentation_says_what_to_run(experiment, capsys):
    assert main(["umap", "-c", str(experiment)]) == 1
    assert "Run 'segment' first" in capsys.readouterr().err

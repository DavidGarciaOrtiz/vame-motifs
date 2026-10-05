"""Running on a GPU server: the ssh/rsync command lines (no server or window needed)."""
import shlex

import pytest
import yaml

from vame_motifs import remote
from vame_motifs.remote import RemoteError, RemoteSettings, remote_job

VALUES = {"pose_files": "pose/", "videos": "", "output": "outputs"}


@pytest.fixture
def server():
    return RemoteSettings(host="me@gpu.example.org", folder="~/vame/exp1", python="~/envs/vame/bin/python")


def test_job_uploads_runs_and_downloads(experiment, server):
    steps, notes = remote_job(server, experiment, "train", ["-v"], VALUES)
    assert [s.kind for s in steps] == ["upload", "upload", "run", "download"]
    assert notes == []

    yaml_step, pose_step, run, download = steps
    assert yaml_step.stdin == experiment.read_text()  # paths inside the folder: sent unchanged
    assert yaml_step.args[-1] == 'mkdir -p "$HOME"/vame/exp1 && cat > "$HOME"/vame/exp1/experiment.yaml'
    assert pose_step.args[-2:] == [f"{experiment.parent / 'pose'}/", "me@gpu.example.org:vame/exp1/pose/"]

    assert run.tty and "-tt" in run.args and "BatchMode=yes" in run.args
    assert run.args[-1] == ('cd "$HOME"/vame/exp1 && export PYTHONUNBUFFERED=1 && '
                            "~/envs/vame/bin/python -m vame_motifs train -c experiment.yaml -v")
    assert download.args[-2:] == ["me@gpu.example.org:vame/exp1/outputs/", f"{experiment.parent / 'outputs'}/"]
    assert download.makedir == experiment.parent / "outputs"


def test_connection_options_go_through_the_tunnel(experiment, server):
    server.port, server.jump_host, server.prefix = "2222", "me@login.example.org", "srun --gres=gpu:1"
    steps, _ = remote_job(server, experiment, "run", [], VALUES)
    run = steps[2].args
    assert run[run.index("-p") + 1] == "2222" and run[run.index("-J") + 1] == "me@login.example.org"
    assert f"ControlPath={remote.CONTROL_PATH}" in run
    assert "&& srun --gres=gpu:1 ~/envs/vame/bin/python -m vame_motifs run" in run[-1]
    # rsync uses the same ssh options, so it goes through the same connection
    ssh_for_rsync = shlex.split(steps[1].args[steps[1].args.index("-e") + 1])
    assert ssh_for_rsync == ["ssh", *run[1:run.index("-tt")]]


def test_server_paths_and_no_sync(experiment, server):
    values = dict(VALUES, pose_files="/pool01/data/pose", output="/pool01/results")
    steps, notes = remote_job(server, experiment, "segment", [], values)
    assert [s.kind for s in steps] == ["upload", "run"]  # nothing copied either way
    assert any("/pool01/data/pose" in n and "already exist on the server" in n for n in notes)
    assert any("results stay on the server" in n for n in notes)

    server.sync_data = False
    steps, notes = remote_job(server, experiment, "train", [], VALUES)
    assert [s.kind for s in steps] == ["upload", "run"]
    assert "must exist on the server as ~/vame/exp1/pose" in notes[0]


def test_local_files_outside_the_folder_are_copied(experiment, server, tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    (data / "pose").mkdir()
    (data / "pose" / "rat01DLC.csv").write_text("x")
    (data / "rat01.mp4").write_bytes(b"")
    results = tmp_path_factory.mktemp("results")
    values = dict(VALUES, pose_files=str(data / "pose"), videos=str(data / "rat01.mp4"), output=str(results))

    steps, notes = remote_job(server, experiment, "train", [], values)
    assert [s.kind for s in steps] == ["upload", "upload", "upload", "run", "download"]
    yaml_step, pose_step, video_step, _, download = steps

    sent = yaml.safe_load(yaml_step.stdin)
    assert sent["input"]["pose_files"] == "inputs/pose_files"
    assert sent["input"]["videos"] == "inputs/videos/rat01.mp4"
    assert sent["output"] == "outputs" and sent["input"]["fps"] == 30  # the rest is kept
    assert yaml_step.args[-1].startswith('mkdir -p "$HOME"/vame/exp1 "$HOME"/vame/exp1/inputs "$HOME"/vame/exp1/inputs/videos ')

    assert pose_step.args[-2:] == [f"{(data / 'pose').resolve()}/", "me@gpu.example.org:vame/exp1/inputs/pose_files/"]
    assert video_step.args[-2:] == [str((data / "rat01.mp4").resolve()), "me@gpu.example.org:vame/exp1/inputs/videos/"]
    assert download.args[-2:] == ["me@gpu.example.org:vame/exp1/outputs/", f"{results}/"]
    assert sum("is copied to the server" in n for n in notes) == 2


def test_shared_local_folder_is_copied_once(experiment, server, tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    (data / "rat01DLC.csv").write_text("x")
    steps, _ = remote_job(server, experiment, "train", [], dict(VALUES, pose_files=str(data), videos=str(data)))
    assert [s.kind for s in steps] == ["upload", "upload", "run", "download"]
    sent = yaml.safe_load(steps[0].stdin)["input"]
    assert sent["pose_files"] == sent["videos"] == "inputs/pose_files"


def test_nothing_to_download_after_validate(experiment, server):
    steps, _ = remote_job(server, experiment, "validate", [], VALUES)
    assert [s.kind for s in steps] == ["upload", "upload", "run"]


@pytest.mark.parametrize(("change", "message"), [
    ({"host": ""}, "Set the server"),
    ({"folder": ""}, "Set the folder"),
    ({"folder": "my exp"}, "cannot contain spaces"),
    ({"port": "ssh"}, "Port must be a number"),
])
def test_incomplete_settings(experiment, server, change, message):
    for key, value in change.items():
        setattr(server, key, value)
    with pytest.raises(RemoteError, match=message):
        remote_job(server, experiment, "train", [], VALUES)


def test_shell_and_rsync_paths(server):
    assert remote.shell_path("~") == '"$HOME"'
    assert remote.shell_path("/data/my exp") == "'/data/my exp'"
    assert remote.shell_path("exp1") == "exp1"
    assert remote.rsync_target(server, "~/a/b/") == "me@gpu.example.org:a/b/"
    assert remote.rsync_target(server, "/abs/") == "me@gpu.example.org:/abs/"


def test_settings_file_round_trip(tmp_path):
    path = tmp_path / "gui.json"
    assert remote.load_settings(path) == {"server": {}, "folders": {}}
    remote.save_settings({"server": {"host": "h", "unknown": 1}, "folders": {"/a.yaml": "f"}}, path)
    data = remote.load_settings(path)
    assert remote.settings_from_dict(data["server"]) == RemoteSettings(host="h")
    assert data["folders"] == {"/a.yaml": "f"}

"""Running the commands on a GPU server over SSH, for the window (gui.py).

Training is the slow step and wants a GPU, which a laptop usually lacks. With a
server set up, the window runs each command there instead of locally:

    1. upload    the experiment YAML, and the pose files / videos      (ssh + rsync)
    2. run       ssh server 'cd <folder> && python -m vame_motifs <command> -c <yaml>'
    3. download  the output folder, back to where the local YAML points  (rsync)

Paths inside the experiment folder keep the same place on the server, since the
YAML's paths are relative to it. Pose files / videos found anywhere else on this
computer are copied into the server's folder under inputs/, a local output
folder elsewhere is filled from the server's outputs/, and the YAML sent to the
server says so. A path that is not on this computer is taken to be a path on
the server and is left as it is.

"Connect" opens one SSH connection and keeps it open: an OpenSSH ControlMaster,
a tunnel that every later ssh and rsync call goes through. A password or
two-factor code is then asked once, in a small window (askpass.py), not at every
step. The connection can go through a login node (jump host, ``ssh -J``). With
key-based login, connecting first is optional.

Only the ``ssh`` and ``rsync`` programs are needed locally (both ship with macOS
and Linux); the server needs vame-motifs installed.

Nothing here touches Tk: these functions only build command lines, so they can
be tested without a window or a server.
"""

from __future__ import annotations

import json
import os
import posixpath
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

SETTINGS_FILE = Path.home() / ".config" / "vame-motifs" / "gui.json"

# One socket per user and server (%C: a hash of host, port and user). Unix
# sockets have a ~100-character path limit, so not under the long macOS $TMPDIR.
CONTROL_PATH = None if sys.platform == "win32" else f"/tmp/vame-motifs-{os.getuid()}-%C"

SERVER_OUTPUT = "outputs"  # on the server, for a local output folder outside the experiment folder

# Commands that write nothing to the output folder: nothing to download.
NO_OUTPUT_COMMANDS = ("validate", "status")

SSH_FAILED = 255  # ssh's own exit code when it cannot connect


class RemoteError(ValueError):
    """The server settings are incomplete or invalid."""


@dataclass
class RemoteSettings:
    host: str = ""        # user@server, or a Host name from ~/.ssh/config
    port: str = ""        # empty: 22, or what ~/.ssh/config says
    key_file: str = ""    # empty: ssh's default keys
    jump_host: str = ""   # login node to tunnel through (ssh -J), optional
    folder: str = ""      # the experiment folder on the server; relative = in the home folder
    python: str = "python"  # the server's Python with vame-motifs installed
    prefix: str = ""      # put before the command, e.g. 'srun --gres=gpu:1' on Slurm
    sync_data: bool = True  # copy pose files/videos up and results down

    def check(self) -> None:
        if not self.host.strip():
            raise RemoteError("Set the server (user@host) in the 'GPU server' tab.")
        if not self.folder.strip():
            raise RemoteError("Set the folder on the server in the 'GPU server' tab.")
        if any(c.isspace() for c in self.folder.strip()):
            raise RemoteError("The folder on the server cannot contain spaces.")
        if self.port.strip() and not self.port.strip().isdigit():
            raise RemoteError(f"Port must be a number (got '{self.port}').")
        if not self.python.strip():
            raise RemoteError("Set the Python on the server in the 'GPU server' tab.")


@dataclass
class Step:
    """One process of a job: run in order, output shown in the window."""
    kind: str             # 'upload', 'run' or 'download'
    title: str            # shown in the log before the step starts
    args: list[str]
    cwd: Path | None = None
    stdin: str | None = None  # text fed to the process
    tty: bool = False     # keep stdin open (ssh -tt): stopping the job ends the remote command
    makedir: Path | None = None  # local folder created before the step


# ======================================================================
# Settings file (per user, not part of the experiment YAML)
# ======================================================================
def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """{'server': {...}, 'folders': {local yaml path: folder on server}}; empty if missing or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"server": {}, "folders": {}}
    return {"server": dict(data.get("server") or {}), "folders": dict(data.get("folders") or {})}


def save_settings(data: dict, path: Path = SETTINGS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def settings_from_dict(values: dict) -> RemoteSettings:
    known = asdict(RemoteSettings())
    return RemoteSettings(**{key: values[key] for key in known if key in values})


def default_folder(yaml_path: Path) -> str:
    """A folder in the server's home named after the local experiment folder."""
    return f"vame-motifs/{yaml_path.parent.name}"


# ======================================================================
# Command lines
# ======================================================================
def _ssh_options(s: RemoteSettings) -> list[str]:
    opts = ["-o", "ServerAliveInterval=30"]  # long training: notice a dead connection
    if s.port.strip():
        opts += ["-p", s.port.strip()]
    if s.key_file.strip():
        opts += ["-i", str(Path(s.key_file.strip()).expanduser())]
    if s.jump_host.strip():
        opts += ["-J", s.jump_host.strip()]
    if CONTROL_PATH:
        opts += ["-o", f"ControlPath={CONTROL_PATH}"]
    return opts


def _client_options(s: RemoteSettings) -> list[str]:
    # BatchMode: never wait for a password nobody can type; use the tunnel or a key.
    return [*_ssh_options(s), "-o", "ControlMaster=no", "-o", "BatchMode=yes"]


def ssh_command(s: RemoteSettings, remote: str, tty: bool = False) -> list[str]:
    return ["ssh", *_client_options(s), *(["-tt"] if tty else []), s.host.strip(), remote]


def master_command(s: RemoteSettings) -> list[str]:
    """The tunnel: logs in once (asking for a password if needed), runs nothing, stays open."""
    return ["ssh", *_ssh_options(s), "-o", "ControlMaster=yes", "-o", "ControlPersist=no", "-N", s.host.strip()]


def check_command(s: RemoteSettings) -> list[str]:
    """Exit code 0 when the tunnel is open and ready."""
    return ["ssh", *_ssh_options(s), "-O", "check", s.host.strip()]


def exit_command(s: RemoteSettings) -> list[str]:
    return ["ssh", *_ssh_options(s), "-O", "exit", s.host.strip()]


def shell_path(path: str) -> str:
    """A server path for the remote shell, quoted, with '~' still meaning the home folder."""
    path = path.strip()
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def rsync_target(s: RemoteSettings, path: str) -> str:
    """host:path for rsync; rsync reads relative server paths from the home folder."""
    path = path.strip()
    if path == "~":
        path = "."
    elif path.startswith("~/"):
        path = path[2:]
    return f"{s.host.strip()}:{path}"


def _rsync(s: RemoteSettings, source: str, dest: str) -> list[str]:
    return ["rsync", "-az", "-e", shlex.join(["ssh", *_client_options(s)]), source, dest]


def _inside(text: str) -> str | None:
    """A YAML path as a clean path relative to the experiment folder, or None if it is
    absolute (a server path) or points outside the folder."""
    text = text.strip()
    if not text or Path(text).expanduser().is_absolute():
        return None
    rel = os.path.normpath(text)
    return None if rel == ".." or rel.startswith(".." + os.sep) else rel


def _server_yaml(yaml_path: Path, server_paths: dict[str, str]) -> str:
    """The experiment YAML as the server should read it: ``server_paths`` replace
    the input/output paths that point to local copies."""
    text = yaml_path.read_text()
    if not server_paths:
        return text
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise RemoteError(f"{yaml_path.name} is not valid YAML: {e}") from e
    for key, path in server_paths.items():
        if key == "output":
            raw["output"] = path
        else:
            raw.setdefault("input", {})[key] = path
    return yaml.safe_dump(raw, sort_keys=False, default_flow_style=False)


def remote_job(s: RemoteSettings, yaml_path: Path, command: str, flags: list[str], values: dict) -> tuple[list[Step], list[str]]:
    """The steps that run ``command`` on the server, and notes for the log.

    ``values`` are the form values (gui.form_to_yaml's input): the pose files,
    videos and output paths decide what is copied.
    """
    s.check()
    base, folder = yaml_path.parent, s.folder.strip().rstrip("/")
    notes: list[str] = []

    uploads: dict[Path, str] = {}  # local path -> path on the server, relative to folder
    server_paths: dict[str, str] = {}  # YAML key -> its new value on the server
    for key in ("pose_files", "videos"):
        text = values.get(key, "").strip()
        if not text:
            continue
        rel, local = _inside(text), base / Path(text).expanduser()
        if not s.sync_data:
            where = text if rel is None else f"{folder}/{rel}"
            notes.append(f"{key}: not copied ('Copy data and results' is off): it must exist on the server as {where}.")
        elif rel is not None:
            if local.exists():
                uploads[local] = rel
            else:
                notes.append(f"{key}: {local} does not exist here, so it is not copied.")
        elif local.exists():  # elsewhere on this computer: copied into the server's folder
            local = local.resolve()
            if local not in uploads:  # pose files and videos may share a folder
                uploads[local] = f"inputs/{key}" if local.is_dir() else f"inputs/{key}/{local.name}"
            server_paths[key] = uploads[local]
            notes.append(f"{key}: {local} is copied to the server as {folder}/{uploads[local]}.")
        else:
            notes.append(f"{key}: {text} is not on this computer: it must already exist on the server.")

    output = values.get("output", "").strip() or "outputs"
    out_rel, local_out = _inside(output), base / Path(output).expanduser()
    if out_rel is None and s.sync_data and (local_out.exists() or local_out.parent.exists()):
        out_rel = server_paths["output"] = SERVER_OUTPUT  # a local folder elsewhere: filled from the server

    # Create the folders (the experiment folder and each upload's parent), then
    # write the YAML through the same ssh call: no rsync needed for it.
    dirs = dict.fromkeys([folder] + [posixpath.normpath(f"{folder}/{Path(rel).parent.as_posix()}") for rel in uploads.values()])
    mkdir = "mkdir -p " + " ".join(shell_path(d) for d in dirs)
    write_yaml = f"cat > {shell_path(f'{folder}/{yaml_path.name}')}"
    steps = [Step("upload", f"Copying {yaml_path.name} to {s.host}:{folder}",
                  ssh_command(s, f"{mkdir} && {write_yaml}"), stdin=_server_yaml(yaml_path, server_paths))]

    for local, rel in uploads.items():
        if local.is_dir():
            source, dest = f"{local}/", rsync_target(s, f"{folder}/{Path(rel).as_posix()}/")
        else:
            source, dest = str(local), rsync_target(s, f"{folder}/{Path(rel).parent.as_posix()}/")
        steps.append(Step("upload", f"Copying {local} to the server (only what changed)", _rsync(s, source, dest)))

    cli = shlex.join(["-m", "vame_motifs", command, "-c", yaml_path.name, *flags])
    run = f"{s.python.strip()} {cli}"
    if s.prefix.strip():
        run = f"{s.prefix.strip()} {run}"
    remote = f"cd {shell_path(folder)} && export PYTHONUNBUFFERED=1 && {run}"
    steps.append(Step("run", f"[{s.host}] vame-motifs {shlex.join([command, '-c', yaml_path.name, *flags])}",
                      ssh_command(s, remote, tty=True), tty=True))

    if s.sync_data and command not in NO_OUTPUT_COMMANDS:
        if out_rel is None:
            notes.append(f"output: {output} is not on this computer: results stay on the server.")
        else:
            steps.append(Step("download", f"Copying the results back to {local_out}",
                              _rsync(s, rsync_target(s, f"{folder}/{Path(out_rel).as_posix()}/"), f"{local_out}/"),
                              makedir=local_out))
    return steps, notes

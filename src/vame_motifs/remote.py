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

The tools, for a reader who has not used them
---------------------------------------------
- ``ssh user@host 'some command'`` logs into another computer and runs a
  command there; its output appears here as if it were local.
- ``rsync source dest`` copies files; with ``host:path`` on one side it copies
  over ssh. It only sends what changed since the last copy, so re-running a
  command does not re-upload gigabytes of video.
- A *command line* is given to Python's ``subprocess`` as a list of words,
  e.g. ``["ssh", "-p", "2222", "me@gpu", "ls"]``. This module only *builds*
  those lists (as ``Step`` objects); gui.py runs them one after the other.
- *Quoting*: a command sent to the server is read by the server's shell,
  which treats spaces and symbols specially. ``shlex.quote`` wraps a text so
  the shell reads it as one plain word.
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

# Where the window remembers the server settings, per user (not in the experiment).
SETTINGS_FILE = Path.home() / ".config" / "vame-motifs" / "gui.json"

# One socket per user and server (%C: a hash of host, port and user). Unix
# sockets have a ~100-character path limit, so not under the long macOS $TMPDIR.
# (The "socket" is the file through which later ssh calls reach the open tunnel.)
CONTROL_PATH = None if sys.platform == "win32" else f"/tmp/vame-motifs-{os.getuid()}-%C"

SERVER_OUTPUT = "outputs"  # on the server, for a local output folder outside the experiment folder

# Commands that write nothing to the output folder: nothing to download.
NO_OUTPUT_COMMANDS = ("validate", "status")

SSH_FAILED = 255  # ssh's own exit code when it cannot connect


class RemoteError(ValueError):
    """The server settings are incomplete or invalid."""


@dataclass
class RemoteSettings:
    """What the 'GPU server' tab of the window holds. All text, as typed."""

    host: str = ""        # user@server, or a Host name from ~/.ssh/config
    port: str = ""        # empty: 22, or what ~/.ssh/config says
    key_file: str = ""    # empty: ssh's default keys
    jump_host: str = ""   # login node to tunnel through (ssh -J), optional
    folder: str = ""      # the experiment folder on the server; relative = in the home folder
    python: str = "python"  # the server's Python with vame-motifs installed
    prefix: str = ""      # put before the command, e.g. 'srun --gres=gpu:1' on Slurm
    sync_data: bool = True  # copy pose files/videos up and results down

    def check(self) -> None:
        """Raise RemoteError naming the field to fix, if a setting is missing or wrong."""
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
    args: list[str]       # the command line to run
    cwd: Path | None = None  # folder to run it in (None: the window's own)
    stdin: str | None = None  # text fed to the process
    tty: bool = False     # keep stdin open (ssh -tt): stopping the job ends the remote command
    makedir: Path | None = None  # local folder created before the step


# ======================================================================
# Settings file (per user, not part of the experiment YAML)
# ======================================================================
def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """{'server': {...}, 'folders': {local yaml path: folder on server}}; empty if missing or unreadable."""
    try:
        data = json.loads(path.read_text())  # JSON: a plain-text format for dictionaries and lists
    except (OSError, ValueError):  # no file yet, or a damaged one: start empty
        return {"server": {}, "folders": {}}
    return {"server": dict(data.get("server") or {}), "folders": dict(data.get("folders") or {})}


def save_settings(data: dict, path: Path = SETTINGS_FILE) -> None:
    """Write the settings (same shape as load_settings returns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def settings_from_dict(values: dict) -> RemoteSettings:
    """A RemoteSettings from a saved dictionary, ignoring keys it does not know
    (e.g. saved by another version); missing keys keep their defaults."""
    known = asdict(RemoteSettings())  # the field names, with their default values
    # **{...} passes a dictionary as keyword arguments: RemoteSettings(host=..., port=...)
    return RemoteSettings(**{key: values[key] for key in known if key in values})


def default_folder(yaml_path: Path) -> str:
    """A folder in the server's home named after the local experiment folder."""
    return f"vame-motifs/{yaml_path.parent.name}"


# ======================================================================
# Command lines
# ======================================================================
def _ssh_options(s: RemoteSettings) -> list[str]:
    """The ssh options every call shares: port, key, jump host, and the tunnel's socket."""
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
    """Options for the ssh calls that do the work (they use the tunnel; they never open one)."""
    # BatchMode: never wait for a password nobody can type; use the tunnel or a key.
    # ("*list" inside a list literal inserts that list's items.)
    return [*_ssh_options(s), "-o", "ControlMaster=no", "-o", "BatchMode=yes"]


def ssh_command(s: RemoteSettings, remote: str, tty: bool = False) -> list[str]:
    """ssh that runs ``remote`` (a shell command line) on the server.

    ``tty=True`` (-tt) gives the remote command a terminal, so that ending
    the local ssh (the window's Stop) also ends the command on the server.
    """
    return ["ssh", *_client_options(s), *(["-tt"] if tty else []), s.host.strip(), remote]


def master_command(s: RemoteSettings) -> list[str]:
    """The tunnel: logs in once (asking for a password if needed), runs nothing, stays open."""
    # ControlMaster=yes: this ssh becomes the shared connection; -N: run no command.
    return ["ssh", *_ssh_options(s), "-o", "ControlMaster=yes", "-o", "ControlPersist=no", "-N", s.host.strip()]


def check_command(s: RemoteSettings) -> list[str]:
    """Exit code 0 when the tunnel is open and ready."""
    return ["ssh", *_ssh_options(s), "-O", "check", s.host.strip()]


def exit_command(s: RemoteSettings) -> list[str]:
    """Asks the tunnel to close."""
    return ["ssh", *_ssh_options(s), "-O", "exit", s.host.strip()]


def shell_path(path: str) -> str:
    """A server path for the remote shell, quoted, with '~' still meaning the home folder.

    Quoting a path hides its "~" from the shell, so "~/" is written as
    "$HOME"/ (the shell's variable for the home folder) followed by the
    quoted rest. Example: ~/my exp -> "$HOME"/'my exp'.
    """
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
    """rsync from source to dest, over ssh with the same options (so through the tunnel).

    -a: keep the files as they are (dates, folders...); -z: compress while sending.
    """
    return ["rsync", "-az", "-e", shlex.join(["ssh", *_client_options(s)]), source, dest]


def _inside(text: str) -> str | None:
    """A YAML path as a clean path relative to the experiment folder, or None if it is
    absolute (a server path) or points outside the folder."""
    text = text.strip()
    if not text or Path(text).expanduser().is_absolute():
        return None
    # normpath tidies the path: "data/./pose/" -> "data/pose", "a/../b" -> "b".
    rel = os.path.normpath(text)
    # Starting with ".." means it goes up, out of the experiment folder.
    return None if rel == ".." or rel.startswith(".." + os.sep) else rel


def _server_yaml(yaml_path: Path, server_paths: dict[str, str]) -> str:
    """The experiment YAML as the server should read it: ``server_paths`` replace
    the input/output paths that point to local copies."""
    text = yaml_path.read_text()
    if not server_paths:
        return text  # nothing to change: send the file exactly as it is
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

    ``flags``: extra command-line words for the command (e.g. ["--force"]).
    Returns (the steps to run in order, sentences explaining what is copied where).
    """
    s.check()
    base, folder = yaml_path.parent, s.folder.strip().rstrip("/")
    notes: list[str] = []

    # --- 1. Decide what to upload, and where it goes on the server ---
    uploads: dict[Path, str] = {}  # local path -> path on the server, relative to folder
    server_paths: dict[str, str] = {}  # YAML key -> its new value on the server
    for key in ("pose_files", "videos"):
        text = values.get(key, "").strip()
        if not text:
            continue  # not set (videos are optional)
        rel, local = _inside(text), base / Path(text).expanduser()
        if not s.sync_data:
            # Copying is switched off: the data must already be on the server.
            where = text if rel is None else f"{folder}/{rel}"
            notes.append(f"{key}: not copied ('Copy data and results' is off): it must exist on the server as {where}.")
        elif rel is not None:
            # Inside the experiment folder: same relative place on the server.
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
            # Not on this computer at all: assumed to be a path on the server.
            notes.append(f"{key}: {text} is not on this computer: it must already exist on the server.")

    # --- 2. Where the results come back to ---
    output = values.get("output", "").strip() or "outputs"
    out_rel, local_out = _inside(output), base / Path(output).expanduser()
    if out_rel is None and s.sync_data and (local_out.exists() or local_out.parent.exists()):
        out_rel = server_paths["output"] = SERVER_OUTPUT  # a local folder elsewhere: filled from the server

    # --- 3. Upload steps ---
    # Create the folders (the experiment folder and each upload's parent), then
    # write the YAML through the same ssh call: no rsync needed for it.
    # ("cat > file" on the server writes whatever it receives on stdin into the file;
    # the YAML text is given as the step's stdin.)
    dirs = dict.fromkeys([folder] + [posixpath.normpath(f"{folder}/{Path(rel).parent.as_posix()}") for rel in uploads.values()])
    mkdir = "mkdir -p " + " ".join(shell_path(d) for d in dirs)
    write_yaml = f"cat > {shell_path(f'{folder}/{yaml_path.name}')}"
    steps = [Step("upload", f"Copying {yaml_path.name} to {s.host}:{folder}",
                  ssh_command(s, f"{mkdir} && {write_yaml}"), stdin=_server_yaml(yaml_path, server_paths))]

    for local, rel in uploads.items():
        # For rsync, a trailing "/" on a folder means "its contents".
        if local.is_dir():
            source, dest = f"{local}/", rsync_target(s, f"{folder}/{Path(rel).as_posix()}/")
        else:
            source, dest = str(local), rsync_target(s, f"{folder}/{Path(rel).parent.as_posix()}/")
        steps.append(Step("upload", f"Copying {local} to the server (only what changed)", _rsync(s, source, dest)))

    # --- 4. The run step: the same command as locally, in the server's folder ---
    cli = shlex.join(["-m", "vame_motifs", command, "-c", yaml_path.name, *flags])
    run = f"{s.python.strip()} {cli}"
    if s.prefix.strip():
        run = f"{s.prefix.strip()} {run}"  # e.g. "srun --gres=gpu:1 python -m vame_motifs ..."
    # PYTHONUNBUFFERED=1: print each line as soon as it is written, so the log is live.
    remote = f"cd {shell_path(folder)} && export PYTHONUNBUFFERED=1 && {run}"
    steps.append(Step("run", f"[{s.host}] vame-motifs {shlex.join([command, '-c', yaml_path.name, *flags])}",
                      ssh_command(s, remote, tty=True), tty=True))

    # --- 5. The download step ---
    if s.sync_data and command not in NO_OUTPUT_COMMANDS:
        if out_rel is None:
            notes.append(f"output: {output} is not on this computer: results stay on the server.")
        else:
            steps.append(Step("download", f"Copying the results back to {local_out}",
                              _rsync(s, rsync_target(s, f"{folder}/{Path(out_rel).as_posix()}/"), f"{local_out}/"),
                              makedir=local_out))
    return steps, notes

"""Command-line entry point: ``vame-motifs <command> -c <experiment.yaml>``.

This module only parses arguments, calls functions from the other modules,
prints results and returns exit codes. The logic lives elsewhere so tests and
notebooks can call it directly.

Typical use::

    vame-motifs validate -c experiment.yaml     # seconds: check the CSVs and the YAML
    vame-motifs run      -c experiment.yaml     # everything: prepare, train, segment, export, communities, videos
    vame-motifs gif      -c experiment.yaml --session rat01 --length 300

Exit codes: 0 success, 1 a problem with the input or a missing step
(short message, no traceback), 2 wrong command-line arguments.

How a command runs, from start to end
-------------------------------------
1. The ``vame-motifs`` program (installed by pip, see pyproject.toml's
   ``[project.scripts]``) calls ``main()`` below.
2. ``main`` reads the words typed (``argparse``): which command, which YAML.
3. It loads the YAML (config.load_config) into a settings object, ``cfg``.
4. For most commands, it creates a run record folder (export.write_run_record)
   and copies the log messages into its ``run.log``.
5. It calls the command's function (``cmd_<name>``), found in ``COMMANDS``.
6. Expected problems (wrong settings, missing step...) are printed as one
   ``Error:`` line and the exit code is 1. Anything unexpected (a real bug)
   is not caught, so Python prints the full traceback, which helps debugging.

Logging: the other modules report progress with ``logger.info(...)`` instead
of ``print``. ``main`` decides where those messages go (the terminal, and
run.log) and how detailed they are (``-v`` shows the DEBUG ones too).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

from vame_motifs import __version__
from vame_motifs.config import ConfigError, ExperimentConfig, load_config
from vame_motifs.export import ExportError, export_motifs, write_run_record
from vame_motifs.io import PoseFileError
from vame_motifs.pipeline import PipelineError, VamePipeline
from vame_motifs.validation import validate

logger = logging.getLogger("vame_motifs")

# Expected problems: a short message is more useful than a traceback.
USER_ERRORS = (ConfigError, PoseFileError, PipelineError, ExportError)


# ======================================================================
# Commands: each takes (cfg, args)
#   cfg:  the experiment settings (config.ExperimentConfig)
#   args: the parsed command line (args.force, args.session, ...)
# Each prints a short result; the work itself is done by other modules.
# ======================================================================
def cmd_validate(cfg: ExperimentConfig, args) -> None:
    """Check the pose files and the YAML, and print a report (validation.py)."""
    poses, reports = validate(cfg)
    print(f"Body parts ({len(poses[0].bodyparts)}): {', '.join(poses[0].bodyparts)}")
    print(f"Alignment : center={cfg.align_center}, direction={cfg.align_direction}")
    if cfg.exclude:
        print(f"Excluded  : {', '.join(cfg.exclude)}")
    if cfg.videos:
        print(f"Videos    : {len(cfg.videos)} file(s) for motif videos")
    print()
    for r in reports:
        status = "OK" if not r.warnings else "WARNING"
        # {x:<45}: x padded to 45 characters, left-aligned; {x:>8}: 8, right-aligned.
        # This lines the report up as a table.
        print(f"{r.name:<45} {r.n_frames:>8} frames   {status}")
        for w in r.warnings:
            print(f"    - {w}")
    print(f"\n{len(reports)} file(s) valid.")


def cmd_init(cfg: ExperimentConfig, args) -> None:
    """Create VAME's project folder (pipeline.VamePipeline.init)."""
    VamePipeline(cfg).init()
    print(f"VAME project ready at {cfg.output / cfg.project_name}")


def cmd_prepare(cfg: ExperimentConfig, args) -> None:
    """Clean and align the poses, and build the training data."""
    VamePipeline(cfg).prepare()


def cmd_train(cfg: ExperimentConfig, args) -> None:
    """Train VAME's model: the slow step."""
    VamePipeline(cfg).train()


def cmd_segment(cfg: ExperimentConfig, args) -> None:
    """Give every time window a motif. --force recomputes existing labels."""
    VamePipeline(cfg).segment(overwrite=args.force)


def cmd_export(cfg: ExperimentConfig, args) -> None:
    """Write the motif CSVs (export.py)."""
    written = export_motifs(cfg)
    print(f"Wrote {len(written)} file(s) to {cfg.motifs_dir}")


def cmd_communities(cfg: ExperimentConfig, args) -> None:
    """Group the motifs into communities, cutting the tree at advanced.community_cut_tree."""
    VamePipeline(cfg).community()
    print(f"Wrote community groupings under {cfg.output / cfg.project_name / 'results' / 'community_cohort'}")


def cmd_videos(cfg: ExperimentConfig, args) -> None:
    """Cut a short video clip of each motif from the raw videos."""
    VamePipeline(cfg).motif_videos()
    print(f"Wrote motif videos (grouped by community) under {cfg.output / cfg.project_name / 'results'}")


def cmd_umap(cfg: ExperimentConfig, args) -> None:
    """Make the 2-D map of the latent space shown in the window, unless it is already up to date."""
    pipeline = VamePipeline(cfg)
    if pipeline.has_umap() and not args.force:
        print(f"Motif map already made: {pipeline.umap_path} (use --force to redo)")
        return
    print(f"Wrote the motif map to {pipeline.umap()}")


def cmd_gif(cfg: ExperimentConfig, args) -> None:
    """Make a vame.gif animation; its options come from the --session, --start... arguments."""
    path = VamePipeline(cfg).gif(session=args.session, algorithm=args.algorithm, start=args.start,
                                 length=args.length, label=args.label, subtract_background=args.subtract_background)
    print(f"Wrote {path}")


def cmd_run(cfg: ExperimentConfig, args) -> None:
    """Everything in order: check, then VAME's steps (pipeline.run), then the CSVs."""
    cmd_validate(cfg, args)  # fail in seconds, not after preprocessing
    VamePipeline(cfg).run(force=args.force)
    cmd_export(cfg, args)


def cmd_status(cfg: ExperimentConfig, args) -> None:
    """Show which steps are done, by looking at which result files exist. Runs nothing."""
    pipeline = VamePipeline(cfg)
    # One label file is expected per recording and per algorithm.
    n_sessions = len(cfg.pose_files) * len(cfg.algorithms)

    # A function defined inside another: only used here.
    def mark(done: bool) -> str:
        return "done" if done else "-"

    print(f"Experiment : {cfg.yaml_path}")
    print(f"Pose files : {len(cfg.pose_files)}")
    print(f"Project    : {pipeline.project_path} [{'exists' if pipeline.project_path.exists() else 'not created'}]")
    print(f"  prepare  : {mark(pipeline.trainset_path.exists())}")
    print(f"  train    : {mark(pipeline.model_path.exists())}")
    print(f"  segment  : {n_sessions - len(pipeline.missing_labels())}/{n_sessions} label files (n_clusters={cfg.n_clusters})")
    for algorithm in cfg.algorithms:
        n_csv = len(list((cfg.motifs_dir / algorithm).glob("*_motifs.csv")))
        print(f"  export   : {algorithm}: {n_csv}/{len(cfg.pose_files)} motif CSVs")
    print(f"  community: {mark(pipeline.has_community())}")
    if cfg.videos:
        print(f"  videos   : {mark(pipeline.has_motif_videos())}")
    print(f"  umap     : {mark(pipeline.has_umap())}")


# The table of commands. Adding a command = writing its cmd_ function and
# adding one line here; the parser and the help text are built from it.
# name -> (function, help text, keeps a run record?)
# (Commands that only read, like validate and status, keep no record.)
COMMANDS: dict[str, tuple[Callable, str, bool]] = {
    "validate": (cmd_validate, "Check the pose CSVs and the YAML (seconds)", False),
    "init":     (cmd_init,     "Create the VAME project from the pose CSVs", True),
    "prepare":  (cmd_prepare,  "Clean, align and build the training set", True),
    "train":    (cmd_train,    "Train and evaluate the VAME model (slow; GPU)", True),
    "segment":  (cmd_segment,  "Assign a motif to every time window", True),
    "export":   (cmd_export,   "Write motif CSVs + motif usage summary", False),
    "communities": (cmd_communities, "Group motifs with similar transitions into communities", True),
    "videos":   (cmd_videos,   "Cut a short .mp4 per motif, per session, from input.videos, grouped by community", True),
    "umap":     (cmd_umap,     "Map the latent space in 2-D (UMAP), for the window's motif map", True),
    "gif":      (cmd_gif,      "vame.gif: the animal next to its path through the UMAP, as a .gif", True),
    "run":      (cmd_run,      "validate -> prepare -> train -> segment -> export -> communities -> videos", True),
    "status":   (cmd_status,   "Show which steps are done", False),
}


# ======================================================================
# Argument parsing
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    """Describe the command line: which words and options are accepted.

    argparse uses this description to read sys.argv, to print ``--help``, and
    to reject wrong input with a message (and exit code 2).
    """

    parser = argparse.ArgumentParser(
        prog="vame-motifs",
        description="Behavioural motifs over time from DeepLabCut pose CSV files, using VAME.",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # keep the line breaks of the epilog
        # The list of commands shown at the end of --help, built from COMMANDS.
        epilog="commands:\n" + "\n".join(f"  {name:<10} {help_text}" for name, (_, help_text, _) in COMMANDS.items()),
    )

    # A positional argument (no dashes): the first word, which must be a command name.
    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND", help="one of: " + ", ".join(COMMANDS))
    parser.add_argument("-c", "--config", required=True, help="experiment YAML file")
    # store_true: a switch; args.force is True when --force is typed, else False.
    parser.add_argument("--force", action="store_true",
                        help="redo finished steps (run) / overwrite segmentation (segment) / redo the map (umap)")
    # Options only used by the gif command, shown together in --help.
    gif = parser.add_argument_group("gif options")
    gif.add_argument("--session", help="recording to show (default: the first)")
    gif.add_argument("--algorithm", choices=["hmm", "kmeans"], help="segmentation to colour by (default: the first)")
    gif.add_argument("--start", type=int, help="first time window (default: random, from the seed)")
    gif.add_argument("--length", type=int, default=500, help="number of frames (default: 500)")
    gif.add_argument("--label", choices=["motif", "community", "none"], default="community",
                     help="colour the map by (default: community)")
    gif.add_argument("--subtract-background", action="store_true", help="remove the static background from the frames")
    parser.add_argument("-v", "--verbose", action="store_true", help="more detailed logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command and return its exit code (0 ok, 1 input problem, 2 wrong arguments).

    ``argv``: the words after the program name; by default the real command
    line. Tests call ``main(["status", "-c", "x.yaml"])`` directly.
    """
    # Wrong arguments make parse_args print a message and exit with code 2 itself.
    args = build_parser().parse_args(argv)

    # Where log messages go (the terminal) and which ones are shown:
    # INFO and above normally, DEBUG too with -v.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    func, _, keeps_record = COMMANDS[args.command]  # "_": the help text, not needed here
    try:
        cfg = load_config(args.config)
        if keeps_record:
            run_dir = write_run_record(cfg, args.command)
            # A second destination for the log messages: the run's run.log file.
            handler = logging.FileHandler(run_dir / "run.log")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logging.getLogger().addHandler(handler)
            logger.info("Run record: %s", run_dir)
        func(cfg, args)  # run the command

    except USER_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)  # errors to stderr, results to stdout
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

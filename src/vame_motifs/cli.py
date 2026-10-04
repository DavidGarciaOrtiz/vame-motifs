"""Command-line entry point: ``vame-motifs <command> -c <experiment.yaml>``.

This module only parses arguments, calls functions from the other modules,
prints results and returns exit codes. The logic lives elsewhere so tests and
notebooks can call it directly.

Typical use::

    vame-motifs validate -c experiment.yaml     # seconds: check the CSVs and the YAML
    vame-motifs run      -c experiment.yaml     # everything: prepare, train, segment, export

Exit codes: 0 success, 1 a problem with the input or a missing step
(short message, no traceback), 2 wrong command-line arguments.
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
# ======================================================================
def cmd_validate(cfg: ExperimentConfig, args) -> None:
    poses, reports = validate(cfg)
    print(f"Body parts ({len(poses[0].bodyparts)}): {', '.join(poses[0].bodyparts)}")
    print(f"Alignment : center={cfg.align_center}, direction={cfg.align_direction}")
    if cfg.exclude:
        print(f"Excluded  : {', '.join(cfg.exclude)}")
    print()
    for r in reports:
        status = "OK" if not r.warnings else "WARNING"
        print(f"{r.name:<45} {r.n_frames:>8} frames   {status}")
        for w in r.warnings:
            print(f"    - {w}")
    print(f"\n{len(reports)} file(s) valid.")


def cmd_init(cfg: ExperimentConfig, args) -> None:
    VamePipeline(cfg).init()
    print(f"VAME project ready at {cfg.output / cfg.project_name}")


def cmd_prepare(cfg: ExperimentConfig, args) -> None:
    VamePipeline(cfg).prepare()


def cmd_train(cfg: ExperimentConfig, args) -> None:
    VamePipeline(cfg).train()


def cmd_segment(cfg: ExperimentConfig, args) -> None:
    VamePipeline(cfg).segment(overwrite=args.force)


def cmd_export(cfg: ExperimentConfig, args) -> None:
    written = export_motifs(cfg)
    print(f"Wrote {len(written)} file(s) to {cfg.motifs_dir}")


def cmd_run(cfg: ExperimentConfig, args) -> None:
    cmd_validate(cfg, args)  # fail in seconds, not after preprocessing
    VamePipeline(cfg).run(force=args.force)
    cmd_export(cfg, args)


def cmd_status(cfg: ExperimentConfig, args) -> None:
    pipeline = VamePipeline(cfg)
    n_sessions = len(cfg.pose_files) * len(cfg.algorithms)

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


# name -> (function, help text, keeps a run record?)
COMMANDS: dict[str, tuple[Callable, str, bool]] = {
    "validate": (cmd_validate, "Check the pose CSVs and the YAML (seconds)", False),
    "init":     (cmd_init,     "Create the VAME project from the pose CSVs", True),
    "prepare":  (cmd_prepare,  "Clean, align and build the training set", True),
    "train":    (cmd_train,    "Train and evaluate the VAME model (slow; GPU)", True),
    "segment":  (cmd_segment,  "Assign a motif to every time window", True),
    "export":   (cmd_export,   "Write motif CSVs + motif usage summary", False),
    "run":      (cmd_run,      "validate -> prepare -> train -> segment -> export", True),
    "status":   (cmd_status,   "Show which steps are done", False),
}


# ======================================================================
# Argument parsing
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(
        prog="vame-motifs",
        description="Behavioural motifs over time from DeepLabCut pose CSV files, using VAME.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:\n" + "\n".join(f"  {name:<10} {help_text}" for name, (_, help_text, _) in COMMANDS.items()),
    )

    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND", help="one of: " + ", ".join(COMMANDS))
    parser.add_argument("-c", "--config", required=True, help="experiment YAML file")
    parser.add_argument("--force", action="store_true", help="redo finished steps (run) / overwrite segmentation (segment)")
    parser.add_argument("-v", "--verbose", action="store_true", help="more detailed logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    func, _, keeps_record = COMMANDS[args.command]
    try:
        cfg = load_config(args.config)
        if keeps_record:
            run_dir = write_run_record(cfg, args.command)
            handler = logging.FileHandler(run_dir / "run.log")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logging.getLogger().addHandler(handler)
            logger.info("Run record: %s", run_dir)
        func(cfg, args)

    except USER_ERRORS as e:
        print(f"Error: {e}", file=sys.stderr)  # errors to stderr, results to stdout
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# vame-motifs

Behavioural motifs over time from DeepLabCut pose estimations, using [VAME](https://github.com/EthoML/VAME).

**Input:** a folder of DeepLabCut `.csv` files (one per recording) and a short YAML file.
**Output:** one `.csv` per recording with the motif of every frame, plus a motif-usage summary.

No videos, no DeepLabCut installation and no DeepLabCut project are needed: body part
names are read from the CSV header.

Future steps: motif identification video analysis, multi-animal analysis

```
pose .csv files ──▶ validate ──▶ prepare ──▶ train ──▶ segment ──▶ export ──▶ motif .csv per recording
+ experiment.yaml   (seconds)    (align,     (slow,    (hmm and/   (frame,
                                  clean)      GPU)      or kmeans)  time_s, motif)
```

## 1. Install

Python 3.12 or newer. Use a dedicated environment (not conda's `base`):

```bash
conda create -n vame python=3.12
conda activate vame
git clone <repo-url> vame-motifs
cd vame-motifs
python -m pip install .            # or: python -m pip install -e ".[dev]" to develop
vame-motifs --help
```

This installs `vame-py==0.14.4`, the exact VAME version the code is written against.
VAME picks the device by itself: CUDA GPU if present, then Apple MPS, then CPU.

## 2. Prepare your data

- **Pose files:** single-animal DeepLabCut `.csv` files (three header rows:
  `scorer`, `bodyparts`, `coords`). Multi-animal files are not supported yet.
  All files must have the same body parts. The file name (without `.csv`) becomes
  the recording's name in every output.
  
- **Experiment file:** copy [`examples/experiment.yaml`](examples/experiment.yaml)
  next to your data and edit it. Paths in it are relative to the YAML file.

```yaml
input:
  pose_files: data/pose/          # folder of DLC .csv files, or one .csv file
  fps: 30
keypoints:
  align_center: snout             # placed at (0, 0) by egocentric alignment
  align_direction: tailbase       # sets the orientation
  exclude: []                     # body parts to leave out, e.g. [tail_end]
cleaning:
  min_confidence: 0.9             # DLC likelihood below this: point discarded and filled in
motifs:
  n_clusters: 15                  # number of motifs
  method: hmm                     # hmm | kmeans | both
output: outputs
advanced:                         # optional; defaults shown
  max_epochs: 100                 # at least 10
  time_window: 30
  zdims: 30
  seed: 42
```

Don't know the body part names? Run `validate`: it lists them.

## 3. Run

```bash
vame-motifs validate -c experiment.yaml   # check files + YAML, list body parts, warn on poor tracking
vame-motifs run      -c experiment.yaml   # prepare -> train -> segment -> export
```

`run` skips steps that are already done, so it is safe to run again. The steps can also
be run one by one:

| Command   | What it does | Time |
|-----------|--------------|------|
| `validate`| Reads every CSV, checks body parts match and the YAML names exist, reports the share of low-confidence frames per body part | seconds |
| `init`    | Creates the VAME project from the CSVs and writes your settings into VAME's config | seconds |
| `prepare` | Low-confidence cleaning, egocentric alignment, outlier removal, smoothing, training set | seconds–minutes |
| `train`   | Trains and evaluates the VAME model | the slow part: use a GPU |
| `segment` | Assigns a motif to every time window (`--force` recomputes) | minutes (kmeans is much faster than hmm) |
| `export`  | Writes the motif CSVs and the usage summary | seconds |
| `status`  | Shows which steps are done | instant |
| `run`     | `validate` then all of the above (`--force` redoes every step) | |

**Changing settings later**

- New `n_clusters` or `method`: just `vame-motifs run -c experiment.yaml` again. The model is
  reused; only segmentation and export run.
- New `time_window`, `zdims`, `max_epochs`, `min_confidence`, alignment or `exclude`:
  the model must be retrained, so use `vame-motifs run -c experiment.yaml --force`.

Exit codes: `0` success, `1` a problem with the input or a missing step (printed as one
`Error:` line, no traceback), `2` wrong command-line arguments. This makes the
command safe to use in scripts and Slurm jobs.

### On a Slurm cluster (e.g. Pedraforca)

Everything runs on the server; no GUI step is involved. Submit the slow command as a job:

```bash
sbatch --gres=gpu:1 --wrap "conda run -n vame vame-motifs run -c /path/to/experiment.yaml"
```

(Ask the admins for the right partition and GPU options.)

## 4. Outputs

```
outputs/
├── motifs/
│   └── hmm/                         (and/or kmeans/)
│       ├── <recording>_motifs.csv   frame, time_s, motif
│       └── motif_usage.csv          share of time in each motif, one row per recording
├── runs/<date>_<command>/           experiment.yaml copy, versions.txt, run.log
└── vame_project/                    VAME's own project (model, logs, plots in model/evaluate/)
```

`<recording>_motifs.csv`:

```
frame,time_s,motif
...
14,0.4667,
15,0.5,3
16,0.5333,3
```

**Why the first and last frames have no motif.** VAME labels sliding windows of
`time_window` frames, so a recording of N frames gets N − time_window + 1 labels.
Following VAME's own convention (the one its motif videos use), each label belongs to
the centre of its window: label *i* ↔ frame *i* + `time_window // 2`. With
`time_window: 30`, the first 15 and the last 14 frames are left empty. Every row
still matches the frame with the same number in the original video.

`motif_usage.csv` gives, per recording, the fraction of labelled frames spent in each
motif (each row sums to 1).

Each `run`/`train`/... also leaves a run record in `outputs/runs/`, with a copy of the
YAML and the package versions, so any result can be traced back to the exact settings
that produced it.

## 5. Troubleshooting

| Message | Meaning |
|---|---|
| `Unknown body part 'nose'. Available: ...` | Use a name exactly as listed (case matters). |
| `... has body parts [...], but ... has [...]` | All recordings must be tracked with the same DLC body parts. |
| `<part>: 60% of frames below likelihood 0.9` | A warning: that body part will be mostly filled in. Improve tracking in DLC or add it to `keypoints.exclude`. |
| `Training finished without saving a model` | VAME only saves a model after its warm-up epochs; raise `advanced.max_epochs`. |
| `No hmm labels for '...'. Run 'segment' first.` | `export` was run before `segment`, or with a different `n_clusters`. |
| `VAME is not installed in this environment` | Activate the right environment (`conda activate vame`). |

## 6. Code layout

```
src/vame_motifs/
├── cli.py          commands: parse arguments, call functions, print, exit codes
├── config.py       load_config()   experiment YAML -> ExperimentConfig
├── io.py           read_dlc_csv()  DLC CSV -> PoseFile
├── validation.py   validate()      checks before any VAME step (uses config.py + io.py)
├── pipeline.py     VamePipeline    the VAME steps (vame-py 0.14.4)
└── export.py       export_motifs() labels -> motif CSVs + usage summary; run records
```

Only `cli.py` prints. The other modules return data or raise one of four errors
(`ConfigError`, `PoseFileError`, `PipelineError`, `ExportError`), so they can be used
from a notebook or a test without a terminal:

```python
from vame_motifs.config import load_config
from vame_motifs.validation import validate

cfg = load_config("experiment.yaml")
poses, reports = validate(cfg)
```

Tests (no VAME run needed, synthetic DLC files are generated on the fly):

```bash
python -m pip install -e ".[dev]"
pytest
```

## License

VAME is GPL-3.0; this project is distributed under GPL-3.0-or-later.

# vame-motifs

Behavioural motifs over time from DeepLabCut pose estimations, using [VAME](https://github.com/EthoML/VAME).

**Input:** a folder of DeepLabCut `.csv` files (one per recording) and a short YAML file.
**Output:** one `.csv` per recording with the motif of every frame, plus a motif-usage summary.

No DeepLabCut installation or DeepLabCut project is needed: body part names are read
from the CSV header. Raw videos are optional, and only needed for `videos`, the step
that cuts short example clips of each motif.

Future steps: multi-animal analysis

```
pose .csv files ──▶ validate ──▶ prepare ──▶ train ──▶ segment ──▶ export ──▶ motif .csv per recording
+ experiment.yaml   (seconds)    (align,     (slow,    (hmm and/   (frame,                  │
                                  clean)      GPU)      or kmeans)  time_s, motif)           ▼
                                                                                      videos (optional)
                                                                            short .mp4 clips, one per motif
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

- **Videos (optional):** the raw recordings DeepLabCut analysed, only needed for the
  `videos` step. Each video is matched to its pose file the way DeepLabCut names them:
  a video `rat01.mp4` is paired with the pose file `rat01DLC_resnet50_....csv` because
  the video's name is a prefix of the pose file's. `.mp4` and `.avi` are supported.

- **Experiment file:** save the example below as `experiment.yaml` next to your
  data and edit it (or let the window write it for you). Paths in it are relative
  to the YAML file.

```yaml
input:
  pose_files: data/pose/          # folder of DLC .csv files, or one .csv file
  videos: data/videos/            # optional: folder of raw videos, or one file; needed for 'videos'
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
| `communities` | Groups motifs into communities by cutting VAME's motif tree at `advanced.community_cut_tree` | seconds |
| `videos`  | Cuts a short `.mp4` per motif per recording, from `input.videos` (needs `input.videos`), in one folder per community | minutes |
| `umap`    | 2-D map (UMAP) of the latent space, for the window's *Explore results* (`--force` redoes it) | about a minute |
| `gif`     | `vame.gif`: the animal next to its path through the UMAP, as a `.gif` (needs `input.videos`) | under a minute |
| `status`  | Shows which steps are done | instant |
| `run`     | `validate` then all of the above (`--force` redoes every step); `videos` only runs if `input.videos` is set | |

**Changing settings later**

- New `n_clusters` or `method`: just `vame-motifs run -c experiment.yaml` again. The model is
  reused; only segmentation and export run.
- New `time_window`, `zdims`, `max_epochs`, `min_confidence`, alignment or `exclude`:
  the model must be retrained, so use `vame-motifs run -c experiment.yaml --force`.

Exit codes: `0` success, `1` a problem with the input or a missing step (printed as one
`Error:` line, no traceback), `2` wrong command-line arguments. This makes the
command safe to use in scripts and Slurm jobs.

### With the window (GUI)

Everything above can also be done from a window, without the terminal or editing YAML by hand.
It is installed with the package; start it with:

```bash
conda activate vame
vame-motifs-gui                    # empty form
vame-motifs-gui experiment.yaml    # or open an existing experiment file
```

(`python -m vame_motifs.gui` works too.) The window uses Tkinter, which comes with
conda's Python; with a Homebrew Python on macOS you may need `brew install python-tk`.

The window has three parts: the experiment form (left), the commands (right) and their
output (bottom). The bar at the top shows which experiment file is open, with *New*,
*Open…*, *Save* and *Save as…*. A `*` in the title means there are unsaved changes.

**Step by step**

1. **Choose where the experiment file lives.** Click *Save as…* and save
   `experiment.yaml` next to your data (or *Open…* an existing one). Do this first:
   the paths you pick are stored relative to this file.
2. **Data tab.** With *File…* or *Folder…*, choose the pose files (a folder of
   DeepLabCut `.csv` files, or one file) and, optionally, the raw videos (only needed
   for the `videos` step). Set the frame rate and the output folder.
3. **Body parts tab.** Click *Read body parts from pose files*: the names are read from
   the first CSV's header. Pick *Align center* and *Align direction* from the lists, and
   select in *Exclude* any body parts to leave out (click to select or unselect).
4. **Motifs tab.** Set the number of motifs, the method (`hmm`, `kmeans` or `both`) and
   the minimum DLC confidence.
5. **Advanced tab.** Optional; the defaults are the same as in the YAML above. Leave
   them as they are unless you know you need to change them.
6. **Check the data.** Click `validate`. The output panel lists the body parts and warns
   about poorly tracked ones. Fix any `Error:` line (shown in red) before going on.
7. **Run.** Click *Run all*. It runs every step in order and skips steps already done.
   Training is slow: the window stays usable, and the status line at the bottom says
   which command is running. *Stop* ends it.
8. **Look at the results.** When the output panel shows `[Finished]`, click
   *Open output folder* (see [4. Outputs](#4-outputs)). *Status* shows which steps are done.

The form is saved automatically before each command, so what runs is always what you see.

**Buttons and options**

| Control | What it does |
|---|---|
| *Run all* | `vame-motifs run`: all steps, skipping those already done |
| `validate` … `videos` | One step at a time, the same commands as in the table above |
| *Status* | Which steps are done |
| *Force (redo steps)* | Adds `--force` to *Run all* and `segment`: redo the steps instead of skipping them |
| *Verbose* | Adds `-v`: more detailed output |
| *Stop* | Ends the running command (asks first) |
| *Open output folder* | Opens the output folder in Finder / Explorer |
| *Explore results…* | Opens the results window (below); *Watch the videos…* in the Data tab opens it on the videos |
| *Clear* | Empties the output panel |
| *On GPU server* | Runs the commands on the server set in the *GPU server* tab (see below) |

**Explore results**

*Explore results…* opens a second window to look at what went in and what came out,
and to act on it. *Segmentation* (top) picks `hmm` or `kmeans` when the method is `both`.

| Tab | What you can do |
|---|---|
| *Recordings* | Play each input video, with the motif and community of the frame shown under it. *Find* a motif or community and jump from one bout of it to the next |
| *Motifs & communities* | VAME's motif tree (motif circles sized by time spent) next to the UMAP of the latent space. Choose the integer at which to cut the tree (the box, or click the tree at that height): the communities it makes are shown at once. *Apply this cut* saves it as *Community cut tree* and runs `communities`. Click the map to see that moment of the recording |
| *Communities & clips* | Each community and its motifs, with the share of time. Select a motif to play its clip (from `videos`). Name the communities and *Save names*: writes `community_labels.csv` and a `<recording>_motifs_labeled.csv` per recording |
| *GIF* | Make a `vame.gif` for a recording (runs `gif`), and play the GIFs made |

The map is computed once by `umap` (*Make the motif map*), and again after retraining.
Names are kept with the motifs of each community, so a name stays with its group when a
new cut renumbers the communities, and is dropped if that group is split.

**Changing settings later** works as in the terminal: a new number of motifs or method,
just *Run all* again; a change in the *Advanced* tab, alignment, exclusions or minimum
confidence, tick *Force* and *Run all* (the model is retrained).

The window only runs `vame-motifs <command> -c experiment.yaml` for you, so results, run
records and exit codes are identical to the terminal, and an `experiment.yaml` saved
from the window can be used directly with the CLI (for example on the cluster).
It needs a desktop: on a Slurm cluster, use the commands below, or run the window on your
computer and train on the server, as follows.

**Training on a GPU server (SSH)**

Training is slow without a GPU. The window can run the commands on a server you can reach
with SSH, while you keep working on your computer:

1. Install vame-motifs on the server too (section 1), and note the path of that
   environment's Python, e.g. `~/miniconda3/envs/vame/bin/python`.
2. In the **GPU server** tab, fill in:

   | Field | Example | |
   |---|---|---|
   | Server | `me@gpu.example.org` | or a `Host` name from `~/.ssh/config` |
   | Port | | empty = 22 |
   | Key file | `~/.ssh/id_ed25519` | empty = your default SSH key |
   | Jump host | `me@login.example.org` | optional: a login node to go through (`ssh -J`) |
   | Folder on server | `vame-motifs/exp1` | where the experiment goes; relative = in your home folder |
   | Python on server | `~/miniconda3/envs/vame/bin/python` | the Python with vame-motifs installed |
   | Run prefix | `srun --gres=gpu:1 -p gpu` | optional: on a Slurm cluster, to run on a GPU node |

3. Click **Connect**. This opens one SSH connection (through the jump host, if set) and keeps
   it open as a tunnel that every later step goes through, so a password or two-factor code
   is asked only once, in a small window. With an SSH key and no password, connecting first
   is optional.
4. Tick **On GPU server** in the *Run* box and use the buttons as usual.

Each command then runs in three steps, all shown in the output panel:

- **upload**: the experiment file, and, with *Copy pose files and videos…* ticked, the pose
  files and videos from your computer (only what changed, with `rsync`);
- **run**: `vame-motifs <command> -c experiment.yaml` on the server, with its output streamed
  live; *Stop* ends it on the server too;
- **download**: the output folder, back to where your experiment file points, so *Open output
  folder* shows the results (even after a failed command, for its run record).

The pose files, videos and output folder can be anywhere on your computer. Those inside the
experiment folder keep the same place in the server folder; those elsewhere are copied to
`inputs/` in the server folder (and results come back from its `outputs/`), and the
experiment file sent to the server points there. Your local experiment file is not changed.
A path that does not exist on your computer (e.g. `/pool01/...`) is taken to be a path on the
server and is not copied: point it at data already on the server to skip uploading it.
The server settings are saved for your user in `~/.config/vame-motifs/gui.json`, not in the
experiment file, and each experiment remembers its own folder on the server. This needs the
`ssh` and `rsync` programs (included in macOS and Linux); *Connect* is not available on Windows,
where an SSH key is needed instead.

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
│       ├── motif_usage.csv          share of time in each motif, one row per recording
│       ├── community_labels.csv     community, label, motifs   (after 'Save names')
│       └── <recording>_motifs_labeled.csv   frame, time_s, motif, community, label
├── gifs/<recording>_<algorithm>_<label>_<start>-<end>.gif   (after 'gif')
├── runs/<date>_<command>/           experiment.yaml copy, versions.txt, run.log
└── vame_project/                    VAME's own project (model, logs, plots in model/evaluate/)
    └── results/
        ├── umap_embedding.npz       the map of 'umap'
        ├── community_cohort/hmm-<n_clusters>/   tree.graphml, tree.png, cohort_community_bag.npy
        └── <recording>/VAME/hmm-<n_clusters>/cluster_videos/
            ├── community_0/<recording>-motif_2.mp4  (only if input.videos was set, after 'videos')
            ├── community_0/<recording>-motif_6.mp4
            └── ...
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
| `No raw videos configured (input.videos). ...` | `videos` was run without setting `input.videos` in the YAML. |
| `input.videos: no video matches pose file '...'` | No video in `input.videos` has a name that is a prefix of that pose file's name (the DeepLabCut naming convention). |
| `VAME is not installed in this environment` | Activate the right environment (`conda activate vame`). |

## 6. Code layout

```
src/vame_motifs/
├── cli.py          commands: parse arguments, call functions, print, exit codes
├── gui.py          window: edits the experiment YAML, runs the cli.py commands
├── remote.py       the ssh/rsync command lines that run them on a GPU server
├── askpass.py      the password window ssh uses when the window connects
├── config.py       load_config()   experiment YAML -> ExperimentConfig
├── io.py           read_dlc_csv()  DLC CSV -> PoseFile
├── validation.py   validate()      checks before any VAME step (uses config.py + io.py)
├── pipeline.py     VamePipeline    the VAME steps (vame-py 0.14.4)
├── export.py       export_motifs() labels -> motif CSVs + usage summary; run records
├── communities.py  cut_tree()      VAME's motif tree -> communities; names -> *_labeled.csv
└── viewer.py       Explorer        the 'Explore results' window (videos, tree, map, names, GIFs)
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

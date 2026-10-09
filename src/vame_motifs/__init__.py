"""vame-motifs: behavioural motifs over time from DeepLabCut pose CSV files, using VAME.

What this package is
--------------------
A *package* is a folder of Python files (*modules*) that belong together. This
file, ``__init__.py``, is the one Python runs first when any part of the
package is imported (``import vame_motifs``). Here it only stores the version.

The big picture, for a reader with no context
---------------------------------------------
1. A researcher films an animal and runs DeepLabCut (DLC) on the video. DLC
   writes a ``.csv`` file with, for every video frame, the x/y position of each
   tracked body part (snout, ears, tail...) and a "likelihood" (how sure DLC is).
2. VAME (Variational Animal Motion Embedding) is a machine-learning library.
   It learns, from those positions, a compact description of movement and
   splits the recording into *motifs*: short, repeated movement patterns
   (e.g. "turning left", "rearing"). Each motif is just a number 0, 1, 2...
3. This package wraps VAME so that a researcher only writes one small YAML
   settings file and runs one command. It produces one ``.csv`` per recording
   saying which motif the animal is in at every frame.

Where to read next (each module explains itself at its top):

- ``cli.py``          the ``vame-motifs`` command: the entry point of everything
- ``config.py``       reads the experiment YAML file
- ``io.py``           reads the DeepLabCut CSV files
- ``validation.py``   checks the input before the slow steps
- ``pipeline.py``     calls VAME, step by step
- ``export.py``       turns VAME's results into readable CSV files
- ``communities.py``  groups motifs into "communities" and stores their names
- ``gui.py``          the desktop window
- ``viewer.py``       the window's "Explore results" part (videos, plots)
- ``remote.py``       running the commands on a GPU server over SSH
- ``askpass.py``      the small password window used by SSH
"""

# The package version. pyproject.toml reads it from here (``version = { attr = ... }``),
# so this line is the single place where the version number is written.
__version__ = "0.1.0"

"""Allows ``python -m vame_motifs ...`` as an alternative to the ``vame-motifs`` command.

How it works: when Python is started with ``-m <package>``, it runs the file
``__main__.py`` inside that package. So ``python -m vame_motifs status -c x.yaml``
runs this file, with ``status -c x.yaml`` left in ``sys.argv`` (the list of
words typed after the program name).

This is useful because it always uses the Python you typed, so the command
works even when the ``vame-motifs`` shortcut is not on the PATH. The window
(gui.py) starts every command this way.
"""
import sys

from vame_motifs.cli import main

# main() reads sys.argv, runs the command and returns a number: the exit code
# (0 = success, 1 = a problem with the input, 2 = wrong arguments).
# sys.exit() ends Python and hands that number to whoever started it (a
# terminal, a script, the window), so they can tell success from failure.
sys.exit(main())

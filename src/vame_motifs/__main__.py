"""Allows ``python -m vame_motifs ...`` as an alternative to the ``vame-motifs`` command."""
import sys

from vame_motifs.cli import main

sys.exit(main())

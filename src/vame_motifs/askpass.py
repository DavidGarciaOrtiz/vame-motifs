"""Password window for ssh (its SSH_ASKPASS program), used by the window's "Connect".

Background
----------
``ssh`` is the program that opens a secure connection to another computer
(here, a GPU server). When it needs a password it normally asks in the
terminal. The window (gui.py) starts ssh *without* a terminal, so ssh cannot
ask there. Instead, ssh supports an environment variable, ``SSH_ASKPASS``: the
path of a program that ssh runs to ask the question for it.

This file is that program. gui.py sets ``SSH_ASKPASS`` to a tiny script that
runs ``python -m vame_motifs.askpass``.

The contract with ssh
---------------------
- ssh runs this with its question as the only argument (a password, a
  two-factor code, or "continue connecting (yes/no)?" for a new server);
- ssh reads the answer from stdout (what this program prints);
- exit code 1 means cancelled.

ssh also sets ``SSH_ASKPASS_PROMPT`` to say what kind of question it is:
'confirm' (only OK/Cancel is needed) or 'none' (just a notice, no answer).
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog


def main(argv: list[str] | None = None) -> int:
    """Show ssh's question in a small window and print the answer. Returns the exit code.

    ``argv``: the arguments; by default the real command-line arguments
    (``sys.argv[1:]``). Tests can pass their own list instead.
    """
    # sys.argv[0] is the program name, so the question is sys.argv[1].
    argv = sys.argv[1:] if argv is None else argv
    prompt = argv[0].strip() if argv else "Password:"
    kind = os.environ.get("SSH_ASKPASS_PROMPT", "")  # 'confirm': yes/no only; 'none': a notice
    if kind == "none":
        return 0  # nothing to answer

    # Tkinter (tk) is Python's built-in window toolkit. Tk() creates the main
    # window; withdraw() hides it, because only the dialog box should appear.
    # "-topmost" keeps the dialog above other windows so it is not missed.
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "confirm":
            # OK -> exit code 0 (accepted), Cancel -> 1 (refused).
            return 0 if messagebox.askokcancel("SSH", prompt, parent=root) else 1
        # A password is hidden with dots while typed; the yes/no question about
        # a new server is not secret, so it is shown as typed.
        secret = "yes/no" not in prompt  # the new-server question is answered in clear text
        answer = simpledialog.askstring("SSH", prompt, show="•" if secret else None, parent=root)
    finally:
        # 'finally' runs whatever happened above (even an early return), so
        # the hidden window is always closed.
        root.destroy()
    if answer is None:
        return 1  # the user pressed Cancel
    print(answer)  # ssh reads the answer from here
    return 0


# This block only runs when the file is started as a program (python -m
# vame_motifs.askpass), not when another module imports it.
if __name__ == "__main__":
    sys.exit(main())

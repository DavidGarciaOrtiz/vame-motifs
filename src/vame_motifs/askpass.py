"""Password window for ssh (its SSH_ASKPASS program), used by the window's "Connect".

ssh runs this with its question as the only argument (a password, a two-factor
code, or "continue connecting (yes/no)?" for a new server) and reads the answer
from stdout. Exit code 1 means cancelled.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    prompt = argv[0].strip() if argv else "Password:"
    kind = os.environ.get("SSH_ASKPASS_PROMPT", "")  # 'confirm': yes/no only; 'none': a notice
    if kind == "none":
        return 0

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "confirm":
            return 0 if messagebox.askokcancel("SSH", prompt, parent=root) else 1
        secret = "yes/no" not in prompt  # the new-server question is answered in clear text
        answer = simpledialog.askstring("SSH", prompt, show="•" if secret else None, parent=root)
    finally:
        root.destroy()
    if answer is None:
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())

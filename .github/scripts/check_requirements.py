"""Fail if any requirements.txt line is not an exact, hash-pinned requirement.

requirements.txt uses pip's hash-checking mode, in which every requirement must
be pinned with `==` and carry at least one `--hash=sha256:...`. A single
unpinned or unhashed line makes the whole install fail, so catch it in CI.

Usage: python check_requirements.py requirements.txt
"""

import re
import sys

PINNED = re.compile(r"[A-Za-z0-9._-]+==[A-Za-z0-9.!+]+(\s+--hash=sha256:[0-9a-f]{64})+")


def main(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    bad = []
    for n, line in enumerate(lines, 1):
        req = line.split("#", 1)[0].strip()
        if req and not PINNED.fullmatch(req):
            bad.append(f"{path}:{n}: not an exact hash-pinned requirement: {req[:80]}")
    for msg in bad:
        print(msg)
    if not bad:
        print(f"ok  {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"))

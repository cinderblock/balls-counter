#!/usr/bin/env python
"""Extract <script> blocks from web.py's embedded HTML and syntax-check them with node.

Usage: python scripts/check_embedded_js.py [path...]
Defaults to src/ball_counter/web.py. Requires node on PATH.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

def render_fstring_stub(js: str) -> str:
    """Render an f-string template like Python would, but stub each
    {interpolation} with the identifier x — valid both as an expression and
    as an object-literal shorthand member — so node can parse the JS."""
    out = []
    i = 0
    while i < len(js):
        c = js[i]
        if c == "{":
            if js.startswith("{{", i):
                out.append("{")
                i += 2
            else:
                depth = 1
                i += 1
                while i < len(js) and depth:
                    if js[i] == "{":
                        depth += 1
                    elif js[i] == "}":
                        depth -= 1
                    i += 1
                out.append("x")
        elif c == "}":
            out.append("}" if js.startswith("}}", i) else "")
            i += 2 if js.startswith("}}", i) else 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


paths = [Path(p) for p in sys.argv[1:]] or [Path("src/ball_counter/web.py")]
errors = []
n_blocks = 0

for path in paths:
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(r"<script>(.*?)</script>", text, re.DOTALL):
        n_blocks += 1
        js = m.group(1)
        line_offset = text[: m.start(1)].count("\n")
        if "{{" in js:
            js = render_fstring_stub(js)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".js", delete=False, encoding="utf-8"
        ) as f:
            f.write(js)
            tmp = Path(f.name)
        try:
            r = subprocess.run(
                ["node", "--check", str(tmp)], capture_output=True, text=True, shell=True
            )
            if r.returncode != 0:
                errors.append(
                    f"{path} <script> block starting at line {line_offset + 1}:\n{r.stderr}"
                )
        finally:
            tmp.unlink(missing_ok=True)

if errors:
    for err in errors:
        print(err, file=sys.stderr)
    sys.exit(1)

print(f"{n_blocks} script block(s): syntax OK")

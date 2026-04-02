#!/usr/bin/env python
"""Check Python file syntax by parsing the AST."""
import ast
import sys

errors = []
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            ast.parse(f.read(), filename=path)
    except SyntaxError as e:
        errors.append(f"{path}: {e}")

if errors:
    for err in errors:
        print(err, file=sys.stderr)
    sys.exit(1)

print(f"{len(sys.argv) - 1} file(s): syntax OK")

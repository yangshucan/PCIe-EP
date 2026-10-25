"""Print deterministic source hashes for local/remote synchronization checks."""

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_DIRECTORIES = ("csrc", "pcie_ep", "tests", "benchmarks", "scripts", "docs")


def source_manifest(root):
    files = {path for directory in SOURCE_DIRECTORIES for path in (root / directory).rglob("*") if path.is_file()}
    files.update(path for path in root.iterdir() if path.is_file())
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if "__pycache__" not in path.parts
        and not path.name.startswith("._")
        and path.name != ".DS_Store"
        and path.suffix not in (".pyc", ".pyo", ".so", ".o")
        and path.name not in ("build.log",)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps({"sha256": source_manifest(args.root.resolve())}, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()

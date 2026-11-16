# Installation

Requires Git and uv. Run commands from the project root after cloning.

```bash
git clone https://github.com/5yearsKim/buddy-zigzag.git buddy_zigzag
cd buddy_zigzag
git submodule update --init third_party/buddy-mlir third_party/zigzag
uv python install 3.13
uv sync --locked
if [ ! -f .env ]; then cp .env.example .env; fi
uv run buddy-zigzag --help
```

`uv.lock` pins Python dependencies; submodule commits pin dependency source.
uv installs CPU Torch and the local ZigZag checkout into `.venv/`.
Python 3.13 must match the major/minor version used to build Buddy's bindings.

Continue with [setup](setup.md), then the [quick start](quick_start.md).
An existing checkout with working native builds can go directly to the quick start.

## Sharing a reproducible checkout

The Buddy scheduling changes and Gemmini JSON exporter currently include local
uncommitted edits. Commit and push them in their dependency forks, then update
Gemmini's pointer in Chipyard and the dependency pointers here before publishing.
A clone receives committed files only.

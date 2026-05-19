# KONTINUUM Core

Pure Python learning engine extracted from [KONTINUUM](https://github.com/Chance-Konstruktion/ha-kontinuum). No Home Assistant dependency — usable from any Python project.

> **Part of the 3-repo family:**
> **kontinuum-core** (this repo, HA-free Python package on PyPI) ·
> [`ha-kontinuum`](https://github.com/Chance-Konstruktion/ha-kontinuum) (full HA Pro integration with UI) ·
> [`ha-kontinuum-lite`](https://github.com/Chance-Konstruktion/ha-kontinuum-lite) (slim HA integration, no UI)

## Installation

```bash
pip install kontinuum-core
```

## Usage

```python
from kontinuum_core import KontinuumEngine

engine = KontinuumEngine()
snapshot = engine.observe({"token": "bedroom.light.on", "room": "bedroom"})
print(snapshot.surprise)
```

## License

AGPL-3.0 – see LICENSE file.

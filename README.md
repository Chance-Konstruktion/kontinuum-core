# KONTINUUM Core

Pure Python learning engine extracted from [KONTINUUM](https://github.com/Chance-Konstruktion/ha-kontinuum).

Looking for a lighter variant? See [KONTINUUM Lite](https://github.com/Chance-Konstruktion/kontinuum-lite).

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

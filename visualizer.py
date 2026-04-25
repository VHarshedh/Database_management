"""OpenEnv Datacenter (SOC Duel) run visualizer.

Replaces the old chess/Stockfish visualizer. This visualizer reads OpenEnv
JSON logs written by the orchestrators/duel runners and plots:

- Threat over time
- Reward bucket components: Outcome / Integrity / Stealth

Usage:

```bash
python Datacenter-workload_migration/visualizer.py path/to/run.json
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional


def _extract_steps(data: dict[str, Any]) -> list[dict[str, Any]]:
    # Support both shapes:
    # - {"steps": [...]} (common)
    # - {"metadata": {"steps": [...]}} (fallback)
    steps = data.get("steps")
    if isinstance(steps, list):
        return [s for s in steps if isinstance(s, dict)]
    md = data.get("metadata", {})
    if isinstance(md, dict) and isinstance(md.get("steps"), list):
        return [s for s in md["steps"] if isinstance(s, dict)]
    return []


def _get_openenv(md: Any) -> Optional[dict[str, Any]]:
    if not isinstance(md, dict):
        return None
    openenv = md.get("openenv")
    return openenv if isinstance(openenv, dict) else None


def _series_from_steps(steps: list[dict[str, Any]]) -> dict[str, list[float]]:
    threat: list[float] = []
    outcome: list[float] = []
    integrity: list[float] = []
    stealth: list[float] = []

    for s in steps:
        md = s.get("metadata", {})
        oe = _get_openenv(md) or md  # allow direct-in-metadata payloads too
        topo = oe.get("topology") if isinstance(oe, dict) else None
        if isinstance(topo, dict) and isinstance(topo.get("threat"), (int, float)):
            threat.append(float(topo["threat"]))

        bucket = oe.get("bucket") if isinstance(oe, dict) else None
        # Prefer defender bucket if present, else take any first.
        b = None
        if isinstance(bucket, dict):
            if "defender" in bucket and isinstance(bucket["defender"], dict):
                b = bucket["defender"]
            else:
                for v in bucket.values():
                    if isinstance(v, dict):
                        b = v
                        break
        if isinstance(b, dict):
            for k, arr in (("outcome", outcome), ("integrity", integrity), ("stealth", stealth)):
                if isinstance(b.get(k), (int, float)):
                    arr.append(float(b[k]))

    return {
        "threat": threat,
        "outcome": outcome,
        "integrity": integrity,
        "stealth": stealth,
    }


def main() -> None:  # pragma: no cover
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py path/to/run.json", file=sys.stderr)
        raise SystemExit(2)

    path = Path(sys.argv[1]).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("run.json must be an object")

    steps = _extract_steps(data)
    series = _series_from_steps(steps)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise SystemExit(f"matplotlib required for plotting: {exc}")

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(series["threat"], label="threat")
    ax[0].set_ylabel("Threat")
    ax[0].legend()
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(series["outcome"], label="outcome")
    ax[1].plot(series["integrity"], label="integrity")
    ax[1].plot(series["stealth"], label="stealth")
    ax[1].set_ylabel("Reward buckets")
    ax[1].set_xlabel("Step")
    ax[1].legend()
    ax[1].grid(True, alpha=0.25)

    fig.suptitle(f"SOC Duel: {path.name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()


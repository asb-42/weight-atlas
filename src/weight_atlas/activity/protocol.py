"""Activity Protocol v1: Versioned stimulus set for Activity Mode.

The fMRI analogy is literal:
- Protocol = Measurement protocol (frozen stimulus set)
- Scanner = Device/Dtype/Torch-Version configuration
- Activity data = Only comparable within same Protocol + Scanner

Source of truth: specs/activity_protocol.v1.json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StateConfig:
    """Configuration for a single stimulus state."""
    name: str
    content: str
    max_len: int
    description: str = ""


@dataclass(frozen=True)
class ActivityProtocol:
    """Versioned activity protocol."""
    version: str
    states: tuple[StateConfig, ...]

    @property
    def protocol_hash(self) -> str:
        """Compute SHA-256 hash of the canonical protocol serialization."""
        data = {
            "version": self.version,
            "states": [
                {"name": s.name, "content": s.content, "max_len": s.max_len}
                for s in self.states
            ],
        }
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get_state(self, name: str) -> StateConfig:
        """Get state configuration by name."""
        for state in self.states:
            if state.name == name:
                return state
        raise ValueError(f"Unknown state: {name!r}. Valid states: {[s.name for s in self.states]}")


def _load_protocol_from_json(path: Path) -> ActivityProtocol:
    """Load protocol from JSON file."""
    with open(path) as f:
        raw = json.load(f)

    states = tuple(
        StateConfig(
            name=s["name"],
            content=s["content"],
            max_len=s["max_len"],
            description=s.get("description", ""),
        )
        for s in raw["states"]
    )

    if not states:
        raise ValueError(f"protocol {path} defines no states")
    if len({s.name for s in states}) != len(states):
        raise ValueError(
            f"protocol {path} has duplicate state names; captures are keyed "
            "by name and would silently overwrite each other"
        )
    if any(s.max_len <= 0 for s in states):
        raise ValueError(f"protocol {path} has a state with max_len <= 0")

    return ActivityProtocol(version=raw["version"], states=states)


# Registry of all protocols
_PROTOCOLS: dict[str, ActivityProtocol] = {}


def _register_protocol(version: str, path: Path) -> None:
    """Register a protocol from a JSON file."""
    _PROTOCOLS[version] = _load_protocol_from_json(path)


# Load built-in protocols. The repo-root specs/ directory only exists in a
# source checkout — in a wheel install the file is not packaged (yet), and a
# silent empty registry would surface far away as "Unknown protocol version".
# Fail loudly at import time instead so the missing artefact is obvious.
_spec_dir = Path(__file__).resolve().parent.parent.parent.parent / "specs"
_v1_path = _spec_dir / "activity_protocol.v1.json"
if _v1_path.exists():
    _register_protocol("v1", _v1_path)
else:  # pragma: no cover - depends on install layout
    raise RuntimeError(
        f"activity protocol registry is empty: {_v1_path} does not exist. "
        "The shipped protocol specs live in <repo>/specs/; run from a source "
        "checkout or reinstall the package."
    )


def load_protocol(version: str = "v1") -> ActivityProtocol:
    """Load a protocol by version.

    Args:
        version: Protocol version (e.g., "v1")

    Returns:
        ActivityProtocol instance

    Raises:
        ValueError: If protocol version is unknown
    """
    if version not in _PROTOCOLS:
        raise ValueError(
            f"Unknown protocol version: {version!r}. "
            f"Available versions: {list(_PROTOCOLS.keys())}"
        )
    return _PROTOCOLS[version]


def list_protocols() -> list[str]:
    """List available protocol versions."""
    return sorted(_PROTOCOLS.keys())

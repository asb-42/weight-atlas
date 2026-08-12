"""Plugin registry for loaders, statistics, and renderers.

Decorators register implementations by string ID. Duplicate IDs raise
ValueError at registration time so misconfigurations surface early.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_loaders: dict[str, Any] = {}
_stats: dict[str, Any] = {}
_renderers: dict[str, Any] = {}


def register_loader(plug_id: str) -> Callable[[F], F]:
    """Decorator registering a loader class under ``plug_id``."""
    def deco(cls: F) -> F:
        if plug_id in _loaders:
            raise ValueError(f"duplicate loader id: {plug_id}")
        _loaders[plug_id] = cls
        return cls
    return deco


def register_stat(plug_id: str) -> Callable[[F], F]:
    """Decorator registering a statistic class under ``plug_id``."""
    def deco(cls: F) -> F:
        if plug_id in _stats:
            raise ValueError(f"duplicate stat id: {plug_id}")
        _stats[plug_id] = cls
        return cls
    return deco


def register_renderer(plug_id: str) -> Callable[[F], F]:
    """Decorator registering a renderer class under ``plug_id``."""
    def deco(cls: F) -> F:
        if plug_id in _renderers:
            raise ValueError(f"duplicate renderer id: {plug_id}")
        _renderers[plug_id] = cls
        return cls
    return deco


def get_loader(plug_id: str) -> Any:
    if plug_id not in _loaders:
        raise KeyError(f"unknown loader: {plug_id}")
    return _loaders[plug_id]


def get_stat(plug_id: str) -> Any:
    if plug_id not in _stats:
        raise KeyError(f"unknown stat: {plug_id}")
    return _stats[plug_id]


def get_renderer(plug_id: str) -> Any:
    if plug_id not in _renderers:
        raise KeyError(f"unknown renderer: {plug_id}")
    return _renderers[plug_id]


def list_loaders() -> list[str]:
    return sorted(_loaders.keys())


def list_stats() -> list[str]:
    return sorted(_stats.keys())


def list_renderers() -> list[str]:
    return sorted(_renderers.keys())


def reset() -> None:
    """Clear all registrations. Intended for tests only."""
    _loaders.clear()
    _stats.clear()
    _renderers.clear()

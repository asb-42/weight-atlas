"""Tests for core/registry.py."""

from __future__ import annotations

import pytest

from weight_atlas.core import registry


def test_register_and_get_loader():
    @registry.register_loader("dummy")
    class Dummy:
        pass

    assert registry.get_loader("dummy") is Dummy
    assert "dummy" in registry.list_loaders()


def test_duplicate_loader_raises():
    @registry.register_loader("dup")
    class A:
        pass

    with pytest.raises(ValueError, match="duplicate loader"):
        @registry.register_loader("dup")
        class B:
            pass


def test_unknown_loader_raises():
    with pytest.raises(KeyError):
        registry.get_loader("nope")


def test_stat_and_renderer_separate():
    @registry.register_stat("s1")
    class S:
        pass

    @registry.register_renderer("r1")
    class R:
        pass

    assert registry.get_stat("s1") is S
    assert registry.get_renderer("r1") is R

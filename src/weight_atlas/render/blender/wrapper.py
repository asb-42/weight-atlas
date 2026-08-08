"""Blender headless renderer plugin (wrapper entry point).

This module re-exports BlenderRenderer from blender_wrapper so the
canonical import path ``weight_atlas.render.blender.wrapper`` resolves.
"""

from weight_atlas.render.blender.blender_wrapper import BlenderRenderer  # noqa: F401

__all__ = ["BlenderRenderer"]

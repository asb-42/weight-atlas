"""Activity Mode ("fMRI"): Capture forward-pass activations over a versioned protocol."""

from weight_atlas.activity.capture import capture_activity
from weight_atlas.activity.protocol import ActivityProtocol, load_protocol

__all__ = ["ActivityProtocol", "load_protocol", "capture_activity"]

"""Pytest conftest – import registering modules so decorators run."""

from weight_atlas.compare.render import DeltaSheet  # noqa: F401 — registers renderer
from weight_atlas.loaders import (
    gguf_loader,  # noqa: F401 — registers loader
    safetensors_loader,  # noqa: F401 — registers loader
)
from weight_atlas.render import (
    fractal,  # noqa: F401 — registers renderer
    matplotlib_sheet,  # noqa: F401 — registers renderer
)
from weight_atlas.render.blender import blender_wrapper  # noqa: F401 — registers renderer
from weight_atlas.stats import (
    norms,  # noqa: F401 — registers stats
    shape_moments,  # noqa: F401 — registers stats
)

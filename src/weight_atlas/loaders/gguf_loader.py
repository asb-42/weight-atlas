"""GGUF loader: mmap, lazy TensorHandles, registry-ID ``gguf``.

Supports MoE models with 3D stacked expert tensors (ffn_*_exps).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from weight_atlas.core.registry import register_loader
from weight_atlas.core.types import TensorHandle
from weight_atlas.loaders.gguf_dequant import dequantize

# GGUF magic number
GGUF_MAGIC = b"GGUF"

# GGUF MoE expert tensor patterns (3D stacked)
_MOE_EXPS_PATTERNS = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]


def _is_gguf(path: Path) -> bool:
    """Check if a file is a GGUF file by reading magic bytes."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == GGUF_MAGIC
    except OSError:
        return False


def _discover_gguf_files(path: Path) -> list[Path]:
    """Return .gguf files for a file or directory.

    Directory mode is recursive and name-sorted: HF snapshots nest shards in
    quant-subdirs (e.g. ``UD-IQ4_XS/…-00001-of-00003.gguf``), and the
    multi-shard loader loop concatenates tensors across the sorted list —
    the metadata-only first shard contributes 0 tensors, which is harmless.
    """
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*.gguf"))
    if not files:
        raise FileNotFoundError(f"no .gguf files under {path}")
    return files


def _is_moe_exps_tensor(name: str) -> bool:
    """Check if a tensor name is a MoE expert tensor (3D stacked)."""
    return any(pattern in name for pattern in _MOE_EXPS_PATTERNS)


class _SharedExpertDequant:
    """Dequantize a 3D stacked GGUF expert tensor exactly once, then slice.

    All per-expert sub-handles of one stacked tensor share a single
    ``_SharedExpertDequant``, so a layer with ``n_experts`` experts performs one
    full dequantization instead of ``n_experts`` (previously each sub-handle
    re-dequantized the whole 3D tensor on every ``load()`` — quadratic for MoE).

    Shape convention: GGUF reports the *logical* shape via ``tensor.shape`` with
    the expert axis LAST (e.g. ``(hidden, hidden, n_experts)``). Quantized
    tensors report a different ``tensor.data.shape`` (the byte layout, e.g.
    ``(n_blocks, ..., block_bytes)``), so we must reshape the dequantized float
    output to the LOGICAL ``tensor.shape`` and slice the last (expert) axis —
    the gguf dequantizer returns elements in logical-shape order.
    """

    def __init__(
        self,
        data: np.ndarray,
        ggml_type: int,
        logical_shape: tuple[int, ...],
        n_children: int,
    ) -> None:
        if len(logical_shape) != 3:
            raise ValueError(
                f"expected a 3D stacked expert tensor, got shape {logical_shape}"
            )
        self._data = data
        self._ggml_type = ggml_type
        self._shape = tuple(logical_shape)
        self._arr: np.ndarray | None = None
        self._lock = threading.Lock()
        self._n_children = n_children
        self._children_done = 0

    def _ensure(self) -> np.ndarray:
        with self._lock:
            if self._arr is None:
                # Pass the mmap-backed array directly: every decoder accepts
                # buffer objects, and .tobytes() would duplicate the full
                # quantized payload (~GB for stacked expert tensors) on the
                # heap per materialization.
                arr = dequantize(self._data, self._ggml_type)
                if arr.size != int(np.prod(self._shape)):
                    raise ValueError(
                        f"dequantized element count {arr.size} does not match logical "
                        f"shape {self._shape}"
                    )
                self._arr = arr.reshape(self._shape).astype(np.float32)
            return self._arr

    def slice(self, expert_id: int) -> np.ndarray:
        """Return the 2D float32 slice for one expert (last axis = expert axis)."""
        return self._ensure()[:, :, expert_id]

    def release_child(self) -> None:
        """Called by each expert sub-handle's ``TensorHandle.clear()``.

        A stacked parent dequantized to float32 can be ~1 GB (256 experts ×
        2048×512); holding every parent for the whole scan would keep the
        entire model in RAM (~4 bytes/param). Once every sub-handle has
        finished, the 3D array is dropped so memory stays proportional to the
        parents currently being processed. The lock guarantees the array is
        never freed while a sub-handle is still using its slice: a handle only
        clears after its statistics are computed, and numpy slice views keep
        the underlying buffer alive even after ``_arr`` is reset to ``None``.
        """
        with self._lock:
            self._children_done += 1
            if self._children_done >= self._n_children:
                self._arr = None


@register_loader("gguf")
class GGUFLoader:
    """Memory-mapped GGUF loader with lazy dequantization.

    Supports MoE models with 3D stacked expert tensors by splitting them
    into per-expert sub-handles.
    """

    format_id = "gguf"

    def source_files(self, path: Path) -> list[Path]:
        """Same discovery as ``open`` — the shards the scan hashes."""
        return _discover_gguf_files(path)

    def open(self, path: Path) -> list[TensorHandle]:
        try:
            from gguf import GGUFReader
        except ImportError as exc:  # pragma: no cover - gguf is a core dep
            raise ImportError(
                "GGUF model scanning requires the 'gguf' package. Install it with "
                "`pip install gguf` (it is a core weight-atlas dependency, so "
                "reinstall the project to get it)."
            ) from exc

        files = _discover_gguf_files(path)
        handles: list[TensorHandle] = []
        # Per-tensor raw block data for row-block streaming (giant tensors:
        # n-gram embedding tables — see docs/2026-09-01_ngram-table-streaming.md).
        self._tensor_blocks: dict[str, tuple[np.ndarray, int]] = {}
        # PLE head segmentation (Flash-Next n-gram table): cumulative bucket
        # offsets from the `ple.head_offsets` KV → 16 contiguous head row
        # ranges; per-head spectra are then free in the streaming pass.
        self.ngram_head_bounds: tuple[str, list[int]] | None = None

        for f in files:
            reader = GGUFReader(str(f))
            for tensor in reader.tensors:
                name = tensor.name
                # Logical shape as reported by gguf (experts LAST for 3D MoE).
                shape = tuple(int(x) for x in tensor.shape)
                ggml_type = tensor.tensor_type.value
                data = tensor.data  # raw bytes / memmap (byte layout for quantized)

                # Check if this is a 3D MoE expert tensor using the LOGICAL shape
                # (tensor.shape), NOT data.shape which is the quantized byte layout.
                if _is_moe_exps_tensor(name) and len(shape) == 3:
                    # Split into per-expert sub-handles sharing one dequantization.
                    # Expert axis is the LAST dim of the logical shape.
                    n_experts = shape[-1]
                    expert_shape = shape[:-1]  # (hidden, hidden)
                    shared = _SharedExpertDequant(data, ggml_type, shape, n_experts)
                    for expert_id in range(n_experts):
                        handles.append(
                            TensorHandle(
                                name=f"{name}[{expert_id}]",
                                shape=expert_shape,
                                dtype=f"ggml_{ggml_type}",
                                loader=self._make_shared_loader(shared, expert_id),
                                expert_id=expert_id,
                                # Once the last expert sub-handle has been cleared, the
                                # shared parent releases its ~1 GB float32 array.
                                on_clear=shared.release_child,
                            )
                        )
                else:
                    if len(shape) == 2:
                        self._tensor_blocks[name] = (data, ggml_type)
                    handles.append(
                        TensorHandle(
                            name=name,
                            shape=shape,
                            dtype=f"ggml_{ggml_type}",
                            loader=self._make_loader(data, ggml_type, shape),
                        )
                    )

            # PLE head offsets KV → contiguous row bounds for the giant table.
            # The KV is an arch-prefixed UINT64 array (`<arch>.ple.head_offsets`,
            # e.g. qwen4exp) that lives in the METADATA shard only (shard 1 of a
            # multi-file GGUF; later shards have no such field) — so every
            # shard's reader gets a chance to provide it. `field.data` holds
            # the part indices of ALL array elements (no type part is skipped:
            # data[0] IS the first offset, verified against real Flash-Next
            # UD-IQ4_XS: [0, 20000003, 40000026, ...]).
            if self.ngram_head_bounds is None:
                for field_name, head_field in reader.fields.items():
                    if not field_name.endswith(".ple.head_offsets"):
                        continue
                    try:
                        offs = [int(head_field.parts[i][0]) for i in head_field.data]
                    except (IndexError, ValueError):
                        continue
                    if offs:
                        self.ngram_head_bounds = ("per_layer_token_embd.weight", offs)
                    break
        return handles

    def iter_row_blocks(
        self, handle: TensorHandle, rows_per_block: int
    ) -> Iterator[np.ndarray] | None:
        """Yield float32 row blocks of a 2-D tensor, dequantizing per block.

        Two verified layouts (2026-09-01, real Flash-Next GGUF — see
        docs/2026-09-01_ngram-table-streaming.md §2):

        - **bucket-packed** (n-gram tables): one data row = one bucket =
          ``type_size / dims_per_bucket`` contiguous blocks forming one
          vector. Detected via ``data.shape[0] == handle.shape[1]`` (GGUF
          dims reversed). Streams the bucket axis (the tall one); cols =
          ``handle.shape[0]``.
        - **standard row-major**: ``data.shape[0] == prod(shape)/block_size``,
          rows = ``shape[0]``.

        Unknown layouts RAISE — a silent None-fallback on a 28.8 GB tensor
        would OOM the scan at exactly the tensor the streaming path exists
        for.
        """
        info = self._tensor_blocks.get(handle.name)
        if info is None:
            return None
        data, ggml_type = info
        from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType(ggml_type)]
        dim_a, dim_b = handle.shape  # file order (GGUF dims reversed)

        from weight_atlas.loaders.gguf_dequant import dequantize

        if data.shape[0] == dim_b and data.shape[1] % type_size == 0:
            # bucket-packed: data row = one bucket vector; cols = dim_a
            rows, cols = dim_b, dim_a

            def _gen_bucket() -> Iterator[np.ndarray]:
                for lo in range(0, rows, rows_per_block):
                    hi = min(lo + rows_per_block, rows)
                    blk = dequantize(np.ascontiguousarray(data[lo:hi]), ggml_type)
                    yield blk.reshape(hi - lo, cols).astype(np.float32, copy=False)

            return _gen_bucket()

        if data.shape[0] * block_size == dim_a * dim_b and dim_b % block_size == 0:
            # standard row-major: rows = dim_a, each row = dim_b weights
            rows, cols = dim_a, dim_b
            blocks_per_row = dim_b // block_size
            if data.shape[0] != rows * blocks_per_row:
                raise ValueError(
                    f"gguf block layout mismatch for {handle.name!r}: "
                    f"data rows {data.shape[0]} != {rows} × {blocks_per_row}"
                )

            def _gen_rowmajor() -> Iterator[np.ndarray]:
                for lo in range(0, rows, rows_per_block):
                    hi = min(lo + rows_per_block, rows)
                    blk = dequantize(
                        np.ascontiguousarray(
                            data[lo * blocks_per_row : hi * blocks_per_row]
                        ),
                        ggml_type,
                    )
                    yield blk.reshape(hi - lo, cols).astype(np.float32, copy=False)

            return _gen_rowmajor()

        raise ValueError(
            f"gguf row-block streaming: unknown data layout for {handle.name!r} "
            f"(handle shape {handle.shape}, data {data.shape}, ggml_type {ggml_type}) — "
            "refusing to fall back to a full 204-GiB-style materialization"
        )

    def _make_loader(
        self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...]
    ) -> Callable[[], np.ndarray]:
        return lambda: self._load_tensor(data, ggml_type, shape)

    def _make_shared_loader(
        self, shared: _SharedExpertDequant, expert_id: int
    ) -> Callable[[], np.ndarray]:
        """Build a zero-arg loader that slices one expert from a shared dequant."""

        def loader() -> np.ndarray:
            return shared.slice(expert_id)

        return loader

    def _load_tensor(self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...]) -> np.ndarray:
        """Materialise a single tensor as float32."""
        # No .tobytes(): decoders accept buffer objects; a full heap copy of
        # the quantized payload per materialization is wasted RAM.
        arr = dequantize(np.ascontiguousarray(data), ggml_type)
        return arr.reshape(shape).astype(np.float32)
